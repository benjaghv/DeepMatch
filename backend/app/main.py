from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from PyPDF2 import PdfReader
import os
import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
from typing import List, Dict
import logging

load_dotenv()

app = FastAPI()

# Configura CORS
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En desarrollo, en producción especifica los dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file_contents: bytes) -> str:
    try:
        from io import BytesIO
        pdf_file = BytesIO(file_contents)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error al leer el PDF: {str(e)}")
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado")

def search_jobs_serpapi(skills: List[str], location: str = "Chile", experience: str = "", previous_roles: List[str] = []):
    """Busca trabajos usando SerpAPI (necesita API key)"""
    import os
    API_KEY = os.getenv('SERPAPI_KEY')
    if not API_KEY:
        return []
    
    # Extraer palabras clave del rol más reciente y la experiencia
    role_keywords = []
    if previous_roles and len(previous_roles) > 0:
        # Tomar el rol más reciente y extraer palabras clave
        latest_role = previous_roles[0].lower()
        role_keywords = [word for word in latest_role.split() if len(word) > 3]
    
    # Construir queries para diferentes búsquedas
    queries = []
    
    # Query 1: Basada en habilidades específicas
    skill_parts = []
    for skill in skills[:3]:
        skill_parts.append(f'"{skill}"')
    if skill_parts:
        queries.append(f'trabajo ({" OR ".join(skill_parts)})')
    
    # Query 2: Basada en el rol más reciente
    if role_keywords:
        role_query = f'trabajo {" ".join(role_keywords)}'
        queries.append(role_query)
    
    # Query 3: Búsqueda general por área profesional
    if experience:
        # Extraer palabras clave de la experiencia
        exp_words = experience.lower().split()
        prof_keywords = [word for word in exp_words if len(word) > 3][:3]
        if prof_keywords:
            queries.append(f'empleo {" ".join(prof_keywords)}')
    
    all_jobs = []
    
    for query in queries:
        params = {
            'engine': 'google_jobs',
            'q': f"{query} {location}",
            'location': location,
            'api_key': API_KEY,
            'hl': 'es',
            'gl': 'cl',
            'chips': 'date_posted:month',  # Mostrar trabajos del último mes
            'start': 0,
            'num': 10
        }
        
        try:
            response = requests.get('https://serpapi.com/search', params=params)
            data = response.json()
            
            if 'error' in data:
                logging.error(f"Error de SerpAPI: {data['error']}")
                continue
            
            for job in data.get('jobs_results', []):
                # Crear un identificador único para el trabajo
                job_id = f"{job.get('company_name', '')}-{job.get('title', '')}"
                
                job_details = {
                    'title': job.get('title', ''),
                    'company': job.get('company_name', ''),
                    'location': job.get('location', location),
                    'link': job.get('related_links', [{}])[0].get('link', '#') if job.get('related_links') else '#',
                    'description': job.get('description', ''),
                    'date': job.get('detected_extensions', {}).get('posted_at', ''),
                    'salary': job.get('detected_extensions', {}).get('salary', ''),
                    'type': job.get('detected_extensions', {}).get('work_type', ''),
                }
                
                # Solo agregar trabajos únicos que tengan título y descripción
                if (job_details['title'] and job_details['description'] and 
                    not any(existing['title'] == job_details['title'] and 
                           existing['company'] == job_details['company'] 
                           for existing in all_jobs)):
                    all_jobs.append(job_details)
            
        except Exception as e:
            logging.error(f"Error con SerpAPI: {str(e)}")
            continue
    
    # Ordenar los trabajos por fecha (más recientes primero)
    all_jobs.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    # Retornar hasta 10 trabajos únicos
    return all_jobs[:10]

@app.post("/api/analyze-cv")
async def analyze_cv(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se envió ningún archivo")
    
    try:
        # Leer el archivo como bytes
        contents = await file.read()
        
        # Verificar que sea un PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
            
        # Extraer texto del PDF
        cv_text = extract_text_from_pdf(contents)
        
        # Lista de modelos a probar en orden de preferencia
        models_to_try = [
            'llama3-70b-8192',  # Modelo más reciente
            'mixtral-8x7b-32768',  # Modelo anterior (por si lo reactivan)
            'llama-3-70b-8192'  # Alternativa similar
        ]
        
        last_error = None
        
        for model in models_to_try:
            try:
                response = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": """Eres un asistente experto en análisis de CVs y recursos humanos. Tu tarea es extraer información clave de CVs en español y estructurarla en formato JSON.
                            
Reglas importantes para el análisis:
1. Para las habilidades (skills):
   - Extrae habilidades profesionales específicas del área
   - Incluye certificaciones, especializaciones y competencias técnicas
   - Mantén términos técnicos en su forma original
   - Busca habilidades en todo el documento, no solo en secciones marcadas como "habilidades"

2. Para la experiencia:
   - Resume la experiencia laboral de forma concisa
   - Menciona el área o sector principal
   - Incluye años de experiencia total si es posible
   - Destaca especializaciones o áreas de expertise

3. IMPORTANTE - Para la educación:
   - BUSCA ACTIVAMENTE cualquier mención de estudios, incluso si no está en una sección específica
   - Considera palabras clave como: "universidad", "instituto", "liceo", "colegio", "carrera", "título"
   - Si encuentras una práctica profesional, debe haber una institución educativa asociada
   - Extrae la información educativa incluso si está mezclada con la experiencia laboral
   - Si no encuentras el año de graduación, usa el año más reciente mencionado o déjalo en blanco
   - Incluye TODOS los niveles educativos encontrados:
     * Educación superior (universidad, instituto)
     * Educación técnica
     * Educación secundaria
     * Cursos y certificaciones

4. Para roles anteriores:
   - Lista TODOS los roles encontrados, incluyendo prácticas
   - Incluye la institución o empresa
   - Ordena del más reciente al más antiguo
   - Para cada práctica profesional, busca y asocia la institución educativa correspondiente

Formato de salida requerido:
{
    "skills": ["skill1", "skill2", ...],
    "experience": "Resumen de experiencia incluyendo área principal y años",
    "education": [
        {
            "title": "Nombre completo del título o carrera",
            "university": "Nombre de la institución educativa",
            "graduation_date": "YYYY-MM-DD",
            "type": "universidad/técnico/certificación"
        }
    ],
    "previous_roles": ["Cargo específico en Empresa/Institución", ...]
}

IMPORTANTE: Si encuentras prácticas profesionales pero no ves información educativa explícita, DEBES buscar en el contexto para encontrar la institución educativa relacionada, ya que las prácticas siempre están asociadas a un programa educativo."""
                        },
                        {
                            "role": "user",
                            "content": f"Analiza este CV y extrae la información solicitada en formato JSON. Asegúrate de buscar y extraer TODA la información educativa, incluso si no está explícitamente en una sección de educación:\n\n{cv_text[:15000]}"
                        }
                    ],
                    model=model,
                    temperature=0.2,  # Reducido para mayor precisión
                    response_format={"type": "json_object"}
                )
                
                # Si llegamos aquí, el modelo funcionó
                analysis = json.loads(response.choices[0].message.content)
                
                # Asegurar que education sea siempre una lista
                if "education" in analysis:
                    if not isinstance(analysis["education"], list):
                        analysis["education"] = [analysis["education"]]
                else:
                    analysis["education"] = []
                
                skills = analysis.get("skills", [])[:5]
                experience = analysis.get("experience", "")
                previous_roles = analysis.get("previous_roles", [])
                jobs = search_jobs_serpapi(skills, experience=experience, previous_roles=previous_roles)
                
                return {
                    "analysis": analysis,
                    "job_recommendations": jobs
                }
                
            except Exception as e:
                last_error = e
                logging.warning(f"Error con el modelo {model}: {str(e)}")
                continue  # Intentar con el siguiente modelo
        
        # Si llegamos aquí, todos los modelos fallaron
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo procesar la solicitud con ningún modelo. Último error: {str(last_error)}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error al procesar el CV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "API de DeepMatch funcionando"}