# V1.1 | 2025-9

import requests
import json
import os
import re
import random
import time
from typing import Optional


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model: Optional[str] = None,
        file_path: str = None,
    ):
        """
        Initialize the LM Studio client
        Args:
            base_url: The base URL of your LM Studio server (default: http://127.0.0.1:1234)
            model: Optional model name to use
            file_path: Optional path to file to load in memory
        """
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.session = requests.Session()  # Reuse session for keep-alive
        self.model = model or self._get_default_model()
        self.file_content = self._load_file(file_path) if file_path else ""

        # Expanded topic pools for maximum diversity
        self.topics = [
            # Tecnología y AI
            "inteligencia artificial en medicina",
            "robótica avanzada",
            "computación cuántica",
            "realidad virtual inmersiva",
            "blockchain y criptomonedas",
            "internet de las cosas",
            "biotecnología moderna",
            "nanotecnología aplicada",
            "energías renovables",
            # Ciencias
            "descubrimientos arqueológicos",
            "exploración espacial",
            "cambio climático",
            "biología marina",
            "genética molecular",
            "neurociencia cognitiva",
            "física de partículas",
            "astronomía extragaláctica",
            "geología planetaria",
            "microbiología extrema",
            # Medicina y Salud
            "terapias génicas innovadoras",
            "medicina personalizada",
            "salud mental digital",
            "telemedicina rural",
            "dispositivos médicos implantables",
            "farmacogenómica",
            "medicina regenerativa",
            "epidemiología global",
            "nutrición funcional",
            # Arte y Cultura
            "arte digital contemporáneo",
            "restauración de patrimonio histórico",
            "música algorítmica",
            "danza experimental",
            "literatura interactiva",
            "cine inmersivo",
            "fotografía conceptual",
            "escultura cinética",
            "performance urbano",
            "arte colaborativo comunitario",
            # Entretenimiento y Medios
            "producción cinematográfica independiente",
            "streaming de contenido original",
            "videojuegos narrativos",
            "podcasting especializado",
            "realidad aumentada recreativa",
            "espectáculos multimedia",
            "documentales investigativos",
            "animación experimental",
            # Deportes y Actividad Física
            "deportes electrónicos profesionales",
            "entrenamiento biomecánico",
            "nutrición deportiva",
            "recuperación atlética avanzada",
            "deportes adaptativos",
            "fitness personalizado",
            "deportes extremos emergentes",
            "análisis de rendimiento deportivo",
            # Educación y Sociedad
            "pedagogía digital inclusiva",
            "educación ambiental práctica",
            "alfabetización mediática",
            "formación profesional especializada",
            "educación intercultural",
            "metodologías colaborativas",
            "aprendizaje experiencial",
            "educación para adultos mayores",
            # Economía y Negocios
            "economía circular sostenible",
            "microfinanzas comunitarias",
            "comercio justo global",
            "emprendimiento social",
            "economía digital rural",
            "cooperativismo moderno",
            "turismo responsable",
            "innovación empresarial",
            # Ciencias Sociales
            "antropología urbana",
            "sociología digital",
            "psicología comunitaria",
            "estudios culturales comparativos",
            "demografía migratoria",
            "lingüística aplicada",
            "criminología preventiva",
            "trabajo social especializado",
            # Medio Ambiente
            "conservación de ecosistemas",
            "agricultura regenerativa",
            "gestión de residuos",
            "biodiversidad urbana",
            "recursos hídricos",
            "forestación inteligente",
            "energía comunitaria",
            "arquitectura bioclimática",
            # Alimentación y Gastronomía
            "gastronomía molecular creativa",
            "agricultura vertical urbana",
            "fermentación artesanal",
            "seguridad alimentaria global",
            "cocina de aprovechamiento",
            "nutrición ancestral",
            "producción de alimentos alternativos",
            "gastronomía intercultural",
            # Moda y Diseño
            "moda sostenible circular",
            "diseño textil innovador",
            "accesorios inteligentes",
            "patronaje inclusivo",
            "materiales biodegradables",
            "diseño industrial ecológico",
            "arquitectura de interiores terapéutica",
            "diseño gráfico experimental",
            # Transporte y Movilidad
            "movilidad urbana sostenible",
            "transporte público inteligente",
            "logística de última milla",
            "vehículos compartidos",
            "infraestructura ciclista",
            "movilidad rural accesible",
            "transporte marítimo ecológico",
            "aviación comercial eficiente",
            # Vivienda y Urbanismo
            "vivienda social innovadora",
            "ciudades inteligentes humanas",
            "regeneración urbana",
            "arquitectura biofílica",
            "espacios públicos inclusivos",
            "gentrificación controlada",
            "desarrollo rural integrado",
            "planificación territorial participativa",
        ]

        self.perspectives = [
            "avances tecnológicos",
            "impacto social",
            "desafíos éticos",
            "innovaciones",
            "implementación empresarial",
            "beneficios para usuarios",
            "retos técnicos",
            "transformación digital",
            "futuro de la industria",
            "casos de uso prácticos",
            "investigación científica",
            "tendencias emergentes",
            "adopción masiva",
        ]

        # Different writing tones for variety
        self.tones = [
            "neutral e informativo - presenta hechos de manera objetiva sin dramatismo",
            "optimista y esperanzador - enfoca los beneficios y oportunidades positivas",
            "analítico y técnico - profundiza en aspectos técnicos con precisión científica",
            "cauteloso y reflexivo - examina tanto beneficios como posibles riesgos de forma equilibrada",
            "educativo y divulgativo - explica conceptos complejos de manera accesible",
            "pragmático y empresarial - se centra en aplicaciones prácticas y viabilidad comercial",
            "investigativo y exploratorio - indaga en desarrollos emergentes con curiosidad científica",
            "crítico constructivo - evalúa limitaciones y desafíos de manera fundamentada",
            "visionario pero realista - imagina posibilidades futuras manteniendo los pies en la tierra",
            "humanístico - se enfoca en el impacto en las personas y la sociedad",
        ]

    def _get_default_model(self):
        models = self.get_available_models()
        if not models:
            raise RuntimeError(
                "No models available. Make sure a model is loaded in LM Studio."
            )
        print(f"Using model: {models[0]}")
        return models[0]

    def _load_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_available_models(self):
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(self.models_url, timeout=10)
            response.raise_for_status()
            models_data = response.json()
            if "data" in models_data:
                return [model["id"] for model in models_data["data"]]
            return []
        except Exception as e:
            print(f"Could not fetch models: {e}")
            return []

    def generate_topic_instruction(self):
        """Generate a random topic, perspective, and tone for variety"""
        topic = random.choice(self.topics)
        perspective = random.choice(self.perspectives)
        tone = random.choice(self.tones)
        return f"""Tema específico: {topic} desde la perspectiva de {perspective}
Tono de escritura: {tone}"""

    def send_message_with_file(
        self, message: str, max_tokens: int = 3000, temperature: float = 0.8
    ) -> Optional[str]:
        """
        Send a text message along with the preloaded file content to the LM Studio model
        """
        if not self.file_content:
            print("Warning: No file content loaded")

        # Generate random topic for this iteration
        topic_instruction = self.generate_topic_instruction()

        combined_message = f"""{message}

{topic_instruction}

IMPORTANTE: El archivo adjunto es SOLO para referencia de formato CSV. 
NO copies los temas, contenidos o ideas específicas del archivo.
DEBES crear contenido completamente original sobre el tema asignado arriba.

Archivo de referencia de formato:
{self.file_content}"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": combined_message}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            response = self.session.post(
                self.chat_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=3600,
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                # Limpieza de texto eficiente
                text = data["choices"][0]["message"]["content"]
                text = re.sub(r"\s+", " ", text).strip()
                return text
            else:
                print("Error: Unexpected response format")
                print(json.dumps(data, indent=2))
                return None
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to LM Studio.")
            return None
        except requests.exceptions.Timeout:
            print("Error: Request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {e}")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON response from server")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


# ================== Ejemplo de uso ==================
client = LMStudioClient(file_path="noticiasIA.csv")

MESSAGE = """Eres un periodista versátil especializado en diversas áreas del conocimiento humano.

TAREA: Generar EXACTAMENTE UNA línea CSV con formato: 0,0,<Body>,<Title>

REGLAS OBLIGATORIAS:
(1) Body y Title completamente en ESPAÑOL
(2) Body: entre 2500-5000 CARACTERES (no palabras)
(3) Body: prosa continua SIN saltos de línea (\\n o \\r). Simula párrafos con DOBLE ESPACIO
(4) Solo texto narrativo: NO tablas, listas, viñetas o código
(5) NO uses comas en Body o Title - usa punto y coma, dos puntos o guiones largos (—)
(6) NO incluyas URLs
(7) Title: corto, informativo, atractivo, en español, sin comas
(8) NO atribuyas citas a personas reales - usa fuentes genéricas como "expertos" o "investigadores"
(9) Crea contenido COMPLETAMENTE ORIGINAL sobre el tema asignado
(10) ADAPTA el tono de escritura según las instrucciones específicas del tema
(11) Escribe sobre el tema asignado como si fuera una noticia actual e importante
(12) VERIFICACIÓN: asegura longitud 2500-5000 caracteres, sin \\n/\\r, una sola línea

IMPORTANTE: 
- Varía el estilo según el tono asignado
- Evita patrones repetitivos en estructura o vocabulario
- Haz que cada noticia parezca de un campo completamente diferente
- NO menciones que es contenido generado o artificial

SALIDA FINAL: Imprime ÚNICAMENTE la línea CSV como 0,0,<Body>,<Title> sin texto adicional."""

RESULT_FILE_PATH = "noticiasIAF.csv"
BATCH_SIZE = 100

buffer = []
TOTAL_ITERS = 1000

MAX_TOPIC_CHANGE_ATTEMPTS = 10
MAX_TONE_CHANGE_ATTEMPTS = 10

# Track used topics and tones to avoid immediate repetition
recent_topics = []
recent_tones = []
MAX_RECENT_TOPICS = 50
MAX_RECENT_TONES = 15

# Time tracking variables
batch_start_time = time.time()
total_start_time = time.time()
generation_times = []

with open(RESULT_FILE_PATH, "a", encoding="utf-8") as resultCSV:
    for i in range(TOTAL_ITERS):
        # Ensure topic diversity by avoiding recently used topics
        attempt_count = 0
        while attempt_count < MAX_TOPIC_CHANGE_ATTEMPTS:
            topic = random.choice(client.topics)
            if topic not in recent_topics or len(recent_topics) >= len(client.topics):
                break
            attempt_count += 1

        # Ensure tone diversity by avoiding recently used tones
        attempt_count = 0
        while attempt_count < MAX_TONE_CHANGE_ATTEMPTS:
            tone = random.choice(client.tones)
            if tone not in recent_tones or len(recent_tones) >= len(client.tones):
                break
            attempt_count += 1

        # Update recent topics list
        if topic in recent_topics:
            recent_topics.remove(topic)
        recent_topics.append(topic)
        if len(recent_topics) > MAX_RECENT_TOPICS:
            recent_topics.pop(0)

        # Update recent tones list
        if tone in recent_tones:
            recent_tones.remove(tone)
        recent_tones.append(tone)
        if len(recent_tones) > MAX_RECENT_TONES:
            recent_tones.pop(0)

        # Time individual generation
        generation_start = time.time()
        response = client.send_message_with_file(MESSAGE)
        generation_end = time.time()

        if response:
            buffer.append(response)
            generation_times.append(generation_end - generation_start)

        # Flush every BATCH_SIZE responses
        if len(buffer) >= BATCH_SIZE:
            resultCSV.write("\n".join(buffer) + "\n")
            buffer.clear()

        if (i + 1) % BATCH_SIZE == 0:
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            # Average time calculations
            if generation_times:
                recent_times = (
                    generation_times[-BATCH_SIZE:]
                    if len(generation_times) >= BATCH_SIZE
                    else generation_times
                )
                avg_per_news = sum(recent_times) / len(recent_times)
                total_elapsed = batch_end_time - total_start_time
                overall_avg = total_elapsed / (i + 1)

                current_tone_short = tone.split(" - ")[0] if " - " in tone else tone
                print(
                    f"Iteración {i + 1} completada - Tema: {topic} | Tono: {current_tone_short}"
                )
                print(
                    f"  Últimos {BATCH_SIZE} en {batch_duration:.1f}s | Promedio: {avg_per_news:.2f}s por noticia"
                )
                print(
                    f"  Promedio general: {overall_avg:.2f}s | ETA: {((TOTAL_ITERS - (i + 1)) * overall_avg) / 3600:.1f}h"
                )
                print("-" * 60)

            # Reset batch timer
            batch_start_time = time.time()

    # Final flush
    if buffer:
        resultCSV.write("\n".join(buffer) + "\n")

# Final statistics
total_end_time = time.time()
total_duration = total_end_time - total_start_time
overall_avg = total_duration / TOTAL_ITERS if generation_times else 0

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"Total de noticias generadas: {TOTAL_ITERS}")
print(
    f"Tiempo total: {total_duration / 3600:.2f} horas ({total_duration / 60:.1f} minutos)"
)
print(f"Promedio por noticia: {overall_avg:.2f} segundos")
if generation_times:
    fastest = min(generation_times)
    slowest = max(generation_times)
    print(f"Más rápida: {fastest:.2f}s | Más lenta: {slowest:.2f}s")
print(f"Noticias por hora: {TOTAL_ITERS / (total_duration / 3600):.0f}")
print("=" * 60)
