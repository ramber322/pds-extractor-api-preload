import re
from datetime import datetime
from typing import List, Any, Optional, Dict
import numpy as np

# =========================
# SENTENCE-BERT FOR SEMANTIC RELEVANCE
# =========================

_sentence_bert_model = None

def get_sentence_bert_model():
    global _sentence_bert_model
    if _sentence_bert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading Sentence-BERT model... (first time only)")
            _sentence_bert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            print("Sentence-BERT model loaded successfully!")
        except ImportError:
            print("sentence-transformers not installed. Falling back to keyword matching.")
            _sentence_bert_model = None
        except Exception as e:
            print(f"Error loading Sentence-BERT: {e}. Falling back to keyword matching.")
            _sentence_bert_model = None
    return _sentence_bert_model


# =========================
# JOB CATEGORY DESCRIPTIONS
# =========================

JOB_CATEGORY_DESCRIPTIONS = {
    "administrative": "Administrative support, office management, clerical work, records management, filing, correspondence, human resources, personnel management, front desk operations, secretarial duties, data entry, document processing, general office administration, procurement support",
    "finance": "Financial management, accounting, budgeting, bookkeeping, auditing, tax preparation, payroll processing, procurement, purchasing, treasury operations, cashier, financial analysis, funds management, expenditure tracking, revenue collection, fiscal management, property assessment",
    "engineering": "Civil engineering, mechanical engineering, electrical engineering, structural design, construction management, infrastructure development, road and bridge construction, waterworks systems, sanitary engineering, plumbing systems, architectural drafting, site surveying, project planning, AutoCAD design, building codes compliance",
    "construction": "Building inspection, construction supervision, structural safety, electrical systems, mechanical systems, plumbing installation, construction foreman, labor management, carpentry, welding, masonry, painting, roofing, quality control, site management, project supervision, safety compliance",
    "communications": "Public information, media relations, journalism, content creation, social media management, press releases, broadcasting, writing, editing, publicity, marketing, advertising, graphic design, video production, photography, corporate communications, public relations",
    "agriculture": "Agriculture, veterinary medicine, animal husbandry, livestock management, poultry farming, crop production, farming, irrigation systems, harvesting, slaughterhouse operations, meat processing, food safety, hygiene inspection, quarantine procedures, aquaculture, dairy farming, pest control, soil management",
    "sports": "Sports development, recreation management, physical education, athletic coaching, games officiating, tournament organization, fitness training, swimming, basketball, volleyball, badminton, track and field, sports regulation, facility management",
    "technical": "Mechanical repair, automotive maintenance, diesel engine repair, electrical systems, plumbing, welding, machinery operation, equipment maintenance, engine repair, transmission repair, hydraulics, pneumatics, industrial maintenance",
    "it": "Information technology, computer systems, software development, programming, coding, networking, cybersecurity, database management, web development, application support, IT support, help desk, desktop support, systems administration, cloud computing, data analytics, DevOps, technical support, troubleshooting, hardware maintenance",
    "planning": "Urban planning, regional development, zoning regulations, environmental planning, GIS mapping, statistical analysis, research, demographic studies, socioeconomic analysis, population studies, housing development, land use planning",
    "social_services": "Social work, community development, outreach programs, welfare services, family support, children and youth services, elderly care, disability services, population management, health services, nutrition programs, counseling, psychosocial support"
}


# =========================
# ELIGIBILITY DETECTION - ENHANCED WITH RA 1080
# =========================

# Eligibility hierarchy with priority levels
ELIGIBILITY_HIERARCHY = {
    "ra1080_physician": 5,
    "ra1080_teacher": 5,
    "ra1080_social_worker": 5,
    "ra1080_engineer": 5,
    "ra1080_nurse": 5,
    "ra1080_cpa": 5,
    "ra1080_lawyer": 5,
    "ra1080_professional": 4,
    "professional": 3,
    "ra1080_subprofessional": 2,
    "subprofessional": 2,
    "category_ii": 2,
    "data_encoder": 2,
    "category_iii_iv": 1,
    "none": 1,
    "unknown": 0
}


def detect_eligibility_level(text: str) -> str:
    """
    Detect the eligibility level from text.
    Returns specific RA 1080 types or standard levels.
    """
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    
    # === CHECK FOR SPECIFIC RA 1080 ===
    if "ra 1080 (physician)" in text_lower or "ra 1080 (medical doctor)" in text_lower:
        return "ra1080_physician"
    if "ra 1080 (professional teacher)" in text_lower or "ra 1080 (teacher)" in text_lower:
        return "ra1080_teacher"
    if "ra 1080 (social worker)" in text_lower:
        return "ra1080_social_worker"
    
    # === CHECK FOR GENERIC RA 1080 ===
    ra_1080_match = re.search(r'ra\s*1080\s*(?:,\s*as\s*amended\s*)?,?\s*(?:\(([^)]+)\))?', text_lower)
    
    if ra_1080_match:
        profession = ra_1080_match.group(1) if ra_1080_match.group(1) else ""
        
        if not profession:
            return "ra1080_professional"
        
        profession = profession.strip()
        
        two_year_professions = [
            "midwife", "medical laboratory technician", "radiologic technologist",
            "dental technician", "optometrist", "veterinary assistant",
            "caregiver", "practical nurse", "pharmacy assistant"
        ]
        
        four_year_professions = [
            "engineer", "civil engineer", "mechanical engineer", "electrical engineer",
            "physician", "doctor", "nurse", "professional teacher", "teacher",
            "accountant", "cpa", "lawyer", "attorney", "bar passer",
            "social worker", "psychologist", "chemist", "agriculturist",
            "criminologist", "architect", "geologist", "master mariner",
            "marine engineer", "environmental planner", "real estate service"
        ]
        
        for prof in two_year_professions:
            if prof in profession:
                return "ra1080_subprofessional"
        
        for prof in four_year_professions:
            if prof in profession:
                if "physician" in prof or "doctor" in prof:
                    return "ra1080_physician"
                if "teacher" in prof:
                    return "ra1080_teacher"
                if "social worker" in prof:
                    return "ra1080_social_worker"
                return "ra1080_professional"
        
        return "ra1080_professional"
    
    # === CHECK FOR STANDARD CSC ELIGIBILITIES ===
    if "career service professional" in text_lower or "civil service professional" in text_lower or "csc professional" in text_lower:
        return "professional"
    
    if "career service subprofessional" in text_lower or "civil service subprofessional" in text_lower or "csc subprofessional" in text_lower:
        return "subprofessional"
    
    # === CHECK FOR SPECIAL ELIGIBILITIES ===
    if "data encoder" in text_lower and ("mc 6" in text_lower or "mc6" in text_lower):
        return "data_encoder"
    
    if "cat ii" in text_lower or "category ii" in text_lower:
        return "category_ii"
    
    if "cat iii" in text_lower or "category iii" in text_lower or "cat iv" in text_lower or "category iv" in text_lower:
        return "category_iii_iv"
    
    # === CHECK FOR NONE ===
    if "none required" in text_lower or "mc 10" in text_lower or "mc 11" in text_lower:
        return "none"
    
    # === FALLBACK ===
    if any(word in text_lower for word in ["eligible", "eligibility", "civil service", "career service"]):
        return "professional"
    
    return "unknown"


def get_eligibility_priority(eligibility_level: str) -> int:
    """Get the priority/level of an eligibility for comparison"""
    return ELIGIBILITY_HIERARCHY.get(eligibility_level, 0)


def is_eligibility_compatible(applicant_level: str, job_level: str) -> bool:
    """
    Check if applicant's eligibility is compatible with job requirement.
    """
    specific_ra1080_types = [
        "ra1080_physician", "ra1080_teacher", "ra1080_social_worker",
        "ra1080_engineer", "ra1080_nurse", "ra1080_cpa", "ra1080_lawyer"
    ]
    
    if job_level in specific_ra1080_types:
        if applicant_level == job_level:
            return True
        return False
    
    if job_level == "ra1080_professional":
        if applicant_level in specific_ra1080_types:
            return True
        if applicant_level == "ra1080_professional":
            return True
        return False
    
    if job_level == "professional":
        if applicant_level == "professional":
            return True
        if applicant_level in specific_ra1080_types or applicant_level == "ra1080_professional":
            return True
        return False
    
    if job_level == "subprofessional":
        if applicant_level in ["subprofessional", "professional", "ra1080_professional", "ra1080_subprofessional"]:
            return True
        if applicant_level in specific_ra1080_types:
            return True
        return False
    
    if job_level in ["none", "category_iii_iv"]:
        return True
    
    return False


def get_eligibility_match_status(applicant_level: str, job_level: str) -> dict:
    """Get detailed match status including explanation - NO EMOJIS"""
    
    if job_level in ["none", "category_iii_iv"]:
        return {
            "matched": True,
            "status": "NOT REQUIRED",
            "message": "No eligibility required for this position",
            "score": 1.0
        }
    
    if applicant_level in ["none", "unknown"]:
        return {
            "matched": False,
            "status": "NOT MET",
            "message": f"Applicant has no eligibility, needs {job_level}",
            "score": 0.0
        }
    
    if is_eligibility_compatible(applicant_level, job_level):
        if applicant_level == job_level:
            return {
                "matched": True,
                "status": "MET",
                "message": f"Applicant has {applicant_level}, matches required {job_level}",
                "score": 1.0
            }
        else:
            return {
                "matched": True,
                "status": "EXCEEDS",
                "message": f"Applicant has {applicant_level}, exceeds required {job_level}",
                "score": 1.0
            }
    else:
        return {
            "matched": False,
            "status": "NOT MET",
            "message": f"Applicant has {applicant_level}, needs {job_level}",
            "score": 0.0
        }


def get_eligibility(text: str) -> int:
    """
    Determine if applicant has ANY eligibility (for scoring).
    Returns: 1 if has eligibility, 0 otherwise
    """
    if not text:
        return 0
    
    text_lower = text.lower()
    
    if any(word in text_lower for word in [
        "career service", "civil service", "ra 1080", "bar passer", 
        "board passer", "eligible", "licensed", "cpa", "registered",
        "professional", "subprofessional", "data encoder", "cat ii"
    ]):
        return 1
    
    return 0


# =========================
# SEMANTIC RELEVANCE FUNCTIONS
# =========================

def is_text_relevant_semantic(text: str, job_category: str, threshold: float = 0.2) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    
    model = get_sentence_bert_model()
    if model is None:
        return is_text_relevant_keyword(text, job_category)
    
    description = JOB_CATEGORY_DESCRIPTIONS.get(job_category.lower(), "")
    if not description:
        return is_text_relevant_keyword(text, job_category)
    
    try:
        embeddings = model.encode([text, description])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"   Similarity: {similarity:.3f} (threshold: {threshold}) - {'RELEVANT' if similarity >= threshold else 'NOT RELEVANT'}")
        return similarity >= threshold
    except Exception as e:
        print(f"Semantic relevance error: {e}. Falling back to keyword matching.")
        return is_text_relevant_keyword(text, job_category)


# =========================
# FALLBACK: KEYWORD MATCHING
# =========================

JOB_CATEGORY_KEYWORDS = {
    "administrative": ["office", "admin", "clerical", "secretarial", "records", "filing", "correspondence", "reception", "front desk", "personnel", "hr", "human resources", "staff", "support", "assistant", "clerks", "typing", "data entry", "document"],
    "finance": ["accounting", "budget", "finance", "bookkeeping", "audit", "tax", "payroll", "procurement", "purchasing", "supply", "cashier", "treasury", "funds", "expenditure", "revenue", "financial", "fiscal", "budgeting", "assessment", "property"],
    "engineering": ["engineering", "civil", "mechanical", "electrical", "structural", "construction", "design", "blueprint", "autocad", "survey", "building", "infrastructure", "roads", "bridges", "waterworks"],
    "construction": ["building", "inspection", "construction", "safety", "structural", "electrical", "mechanical", "plumbing", "foreman", "labor", "carpentry", "welding", "masonry", "painting", "roofing"],
    "communications": ["information", "communications", "media", "public relations", "journalism", "content", "social media", "press release", "broadcast", "writing", "editing", "publicity", "marketing"],
    "agriculture": ["agriculture", "veterinary", "animal", "livestock", "poultry", "crops", "farming", "irrigation", "harvest", "slaughterhouse", "meat", "food safety", "hygiene", "inspection", "quarantine"],
    "sports": ["sports", "recreation", "physical education", "games", "athletics", "fitness", "coaching", "training", "tournament"],
    "technical": ["mechanic", "automotive", "repair", "maintenance", "diesel", "electrical", "plumbing", "welding", "machinery", "equipment"],
    "it": ["software", "hardware", "network", "database", "programming", "coding", "system", "cybersecurity", "web", "application", "python", "java", "sql", "server", "cloud", "devops", "it", "information technology", "tech", "technical support", "help desk", "desktop", "it support", "it specialist", "computer", "computing", "digital", "data", "analytics"],
    "planning": ["planning", "development", "zoning", "urban", "regional", "environmental", "gis", "mapping", "statistics", "research"],
    "social_services": ["social", "community", "outreach", "development", "welfare", "family", "children", "youth", "elderly", "disability"]
}


def is_text_relevant_keyword(text: str, job_category: str, threshold: float = 0.05) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    
    text_lower = text.lower()
    keywords = JOB_CATEGORY_KEYWORDS.get(job_category.lower(), JOB_CATEGORY_KEYWORDS["administrative"])
    
    matches = 0
    for keyword in keywords:
        if keyword in text_lower:
            matches += 1
    
    match_ratio = matches / len(keywords) if keywords else 0
    return match_ratio >= threshold


def is_text_relevant(text: str, job_category: str, threshold: float = 0.2) -> bool:
    if len(text) < 10:
        return is_text_relevant_keyword(text, job_category)
    return is_text_relevant_semantic(text, job_category, threshold)


def extract_training_title(row: List[Any]) -> Optional[str]:
    for cell in row:
        if cell is None:
            continue
        cell_str = str(cell).strip()
        if cell_str and len(cell_str) > 3:
            if re.match(r'^[\d/.-]+$', cell_str):
                continue
            if re.match(r'^\d+\.?\d*\s*$', cell_str):
                continue
            return cell_str
    return None


def extract_hours_from_text(text: str) -> int:
    text = text.lower().strip()
    total_hours = 0
    
    hour_patterns = [
        r'(\d+\.?\d*)\s*(?:hours?|hrs?|h)\b',
        r'(\d+\.?\d*)\s*(?:hour)\b',
    ]
    for pattern in hour_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                total_hours += float(match)
            except ValueError:
                pass
    
    if total_hours == 0:
        number_match = re.match(r'^(\d+\.?\d*)\s*$', text)
        if number_match:
            try:
                hours = float(number_match.group(1))
                if 1 <= hours <= 500:
                    total_hours += hours
            except ValueError:
                pass
    
    day_pattern = r'(\d+)\s*(?:days?|day)\b'
    matches = re.findall(day_pattern, text)
    for match in matches:
        try:
            total_hours += float(match) * 8
        except ValueError:
            pass
    
    return int(total_hours)


def get_education_level(text: str) -> int:
    text = text.lower()
    
    # === MASTER'S (Level 5) ===
    if any(word in text for word in ["master", "masters", "master's", "ms ", "ma ", "m.a.", "m.s.", "mba"]):
        return 5
    
    # === BACHELOR'S (Level 4) ===
    if any(word in text for word in [
        "bachelor", "bachelors", "bachelor's", 
        "bs ", "ba ", "b.s.", "b.a.", "bsc", "b.s.c",
        "bscs", "bsit", "bs cs"
    ]):
        return 4
    
    # === 2-YEAR COLLEGE / ASSOCIATE (Level 3) ===
    if any(word in text for word in ["college", "university", "undergraduate", "associate"]):
        return 3
    
    # === HIGH SCHOOL (Level 2) ===
    if any(word in text for word in [
        "high school", "highschool", 
        "senior high school", "shs", 
        "grade 12", "grade12", "grade 12 graduate",
        "secondary"
    ]):
        return 2
    
    # === ELEMENTARY (Level 1) ===
    if any(word in text for word in ["elementary", "primary"]):
        return 1
    
    # === NONE REQUIRED / NO EDUCATION (Level 0) ===
    if any(word in text for word in ["none required", "no education", "none"]):
        return 0
    
    return 0


def get_training_hours(training_rows: List[List[Any]], job_category: str = "administrative") -> int:
    if not training_rows:
        return 0
    
    total_relevant_hours = 0
    total_hours = 0
    
    training_keywords = ["seminar", "workshop", "course", "training", "program", "conference", "webinar", "certification"]
    header_keywords = ["learning and development", "l&d", "title of learning", "inclusive dates", "number of hours", "type of l&d", "conducted/sponsored", "write in full", "dd/mm/yyyy"]
    
    print("\n" + "="*70)
    print(f"TRAINING RELEVANCE ANALYSIS (Category: {job_category.upper()})")
    print(f"   Using: Semantic Similarity (Sentence-BERT) with threshold: 0.2")
    print("="*70)
    
    for row in training_rows:
        has_real_content = False
        row_values = []
        
        for cell in row:
            if cell is None:
                continue
            cell_str = str(cell).strip()
            if cell_str and cell_str.lower() != "n/a":
                has_real_content = True
                row_values.append(cell_str)
        
        if not has_real_content:
            continue
        
        row_text = " ".join(row_values).lower()
        
        if "n/a" in row_text and len(row_values) <= 2:
            continue
        
        is_header = False
        for keyword in header_keywords:
            if keyword in row_text:
                is_header = True
                break
        if is_header:
            continue
        
        has_training_keyword = any(k in row_text for k in training_keywords)
        has_date = re.search(r'(20\d{2}|19\d{2})', row_text)
        
        hours = 0
        for cell in row:
            if cell is None:
                continue
            cell_str = str(cell).strip()
            
            hour_match = re.search(r'(\d+\.?\d*)\s*(?:hours?|hrs?|h)\b', cell_str, re.IGNORECASE)
            if hour_match:
                try:
                    hours += float(hour_match.group(1))
                except ValueError:
                    pass
            
            if hours == 0:
                number_match = re.match(r'^(\d+\.?\d*)\s*$', cell_str)
                if number_match:
                    try:
                        h = float(number_match.group(1))
                        if 1 <= h <= 500:
                            hours += h
                    except ValueError:
                        pass
        
        if hours == 0:
            hours = extract_hours_from_text(row_text)
        
        if hours > 0:
            total_hours += hours
            title = extract_training_title(row) or "Unknown Training"
            
            is_relevant = is_text_relevant(row_text, job_category, threshold=0.2)
            
            if is_relevant:
                total_relevant_hours += hours
                status = "RELEVANT"
            else:
                status = "IRRELEVANT"
            
            print(f"  {hours}h - {title[:40]:<40} - {status}")
    
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY:")
    print(f"   Total Training Hours: {total_hours}")
    print(f"   Relevant Hours: {total_relevant_hours}")
    print(f"   Irrelevant Hours: {total_hours - total_relevant_hours}")
    print("="*70 + "\n")
    
    return total_relevant_hours


def get_experience_years(experience_rows: List[List[Any]], job_category: str = "administrative") -> int:
    if not experience_rows:
        return 0
    
    total_experience_years = 0
    relevant_years = 0
    current_year = datetime.now().year
    
    print("\n" + "="*70)
    print(f"EXPERIENCE RELEVANCE ANALYSIS (Category: {job_category.upper()})")
    print(f"   Using: Semantic Similarity (Sentence-BERT) with threshold: 0.2")
    print("="*70)
    
    for row in experience_rows:
        if len(row) < 2:
            continue
        
        from_cell = row[0]
        to_cell = row[1]
        
        position_title = ""
        if len(row) >= 3:
            position_title = str(row[2]) if row[2] else ""
        
        department = ""
        if len(row) >= 4:
            department = str(row[3]) if row[3] else ""
        
        experience_text = f"{position_title} {department}"
        
        start_date = None
        end_date = None
        
        if isinstance(from_cell, datetime):
            start_date = from_cell
        else:
            text = str(from_cell)
            years = re.findall(r'(20\d{2}|19\d{2})', text)
            if years:
                try:
                    start_date = datetime(int(years[0]), 1, 1)
                except:
                    pass
        
        if isinstance(to_cell, datetime):
            end_date = to_cell
        elif isinstance(to_cell, str) and "present" in to_cell.lower():
            end_date = datetime.now()
        else:
            text = str(to_cell)
            years = re.findall(r'(20\d{2}|19\d{2})', text)
            if years:
                try:
                    end_date = datetime(int(years[-1]), 1, 1)
                except:
                    pass
        
        if start_date and end_date:
            days = (end_date - start_date).days
            years = days / 365.25
            if 0 < years < 60:
                total_experience_years += int(round(years))
                
                is_relevant = is_text_relevant(experience_text, job_category, threshold=0.15)
                
                if is_relevant:
                    relevant_years += int(round(years))
                    status = "RELEVANT"
                else:
                    status = "IRRELEVANT"
                
                print(f"  {int(round(years))}y - {position_title[:40]:<40} - {status}")
    
    print("\n" + "="*70)
    print(f"EXPERIENCE SUMMARY:")
    print(f"   Total Experience: {total_experience_years} years")
    print(f"   Relevant Experience: {relevant_years} years")
    print(f"   Irrelevant Experience: {total_experience_years - relevant_years} years")
    print("="*70 + "\n")
    
    return relevant_years


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    return text.strip().lower()