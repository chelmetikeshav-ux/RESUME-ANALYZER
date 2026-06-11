import re

# Categorized Skill Database
# Maps category name to a list of tuples: (Skill Name, Pattern/Aliases list)
SKILL_TAXONOMY = {
    "Programming Languages": [
        ("Python", ["python", "py"]),
        ("JavaScript", ["javascript", "js", "ecmascript"]),
        ("TypeScript", ["typescript", "ts"]),
        ("Java", ["java"]),
        ("C++", ["c\\+\\+"]),
        ("C#", ["c#", "c-sharp"]),
        ("C", ["\\bc\\b"]), # Match lonely 'c' with word boundaries
        ("Go", ["\\bgo\\b", "golang"]),
        ("Rust", ["rust"]),
        ("Ruby", ["ruby", "rails"]),
        ("PHP", ["php"]),
        ("Swift", ["swift"]),
        ("Kotlin", ["kotlin"]),
        ("R", ["\\br\\b"]),
        ("SQL", ["sql", "mysql", "postgresql", "sqlite"]),
        ("Scala", ["scala"]),
        ("HTML", ["html", "html5"]),
        ("CSS", ["css", "css3", "sass", "scss", "less"]),
        ("Shell Scripting", ["bash", "shell", "sh", "powershell"]),
        ("Perl", ["perl"]),
        ("MATLAB", ["matlab"])
    ],
    "Frameworks & Libraries": [
        ("React", ["react", "reactjs", "react.js"]),
        ("Angular", ["angular", "angularjs"]),
        ("Vue", ["vue", "vuejs", "vue.js"]),
        ("Svelte", ["svelte"]),
        ("Next.js", ["next\\.js", "nextjs"]),
        ("Node.js", ["node\\.js", "nodejs", "node"]),
        ("Express", ["express", "expressjs"]),
        ("Django", ["django"]),
        ("Flask", ["flask"]),
        ("FastAPI", ["fastapi"]),
        ("Spring Boot", ["spring boot", "spring"]),
        ("ASP.NET", ["asp\\.net", "\\.net", "dotnet"]),
        ("Ruby on Rails", ["ruby on rails", "rails"]),
        ("Laravel", ["laravel"]),
        ("NestJS", ["nestjs"]),
        ("Tailwind CSS", ["tailwind", "tailwindcss"]),
        ("Bootstrap", ["bootstrap"]),
        ("Redux", ["redux", "redux-toolkit"]),
        ("PyTorch", ["pytorch", "torch"]),
        ("TensorFlow", ["tensorflow", "tf"]),
        ("Keras", ["keras"]),
        ("Scikit-Learn", ["scikit-learn", "sklearn"]),
        ("Pandas", ["pandas"]),
        ("NumPy", ["numpy"]),
        ("SciPy", ["scipy"]),
        ("OpenCV", ["opencv"])
    ],
    "Databases & Caching": [
        ("PostgreSQL", ["postgresql", "postgres"]),
        ("MySQL", ["mysql"]),
        ("MongoDB", ["mongodb", "mongo"]),
        ("Redis", ["redis"]),
        ("SQLite", ["sqlite"]),
        ("DynamoDB", ["dynamodb"]),
        ("Cassandra", ["cassandra"]),
        ("Elasticsearch", ["elasticsearch", "elastic"]),
        ("Firebase", ["firebase"]),
        ("Oracle", ["oracle"]),
        ("SQL Server", ["sql server", "mssql"]),
        ("Neo4j", ["neo4j"]),
        ("MariaDB", ["mariadb"])
    ],
    "Cloud & Infrastructure": [
        ("AWS", ["aws", "amazon web services", "ec2", "s3", "rds", "lambda", "fargate"]),
        ("Azure", ["azure", "microsoft azure"]),
        ("Google Cloud", ["gcp", "google cloud", "google cloud platform"]),
        ("Docker", ["docker"]),
        ("Kubernetes", ["kubernetes", "k8s"]),
        ("Terraform", ["terraform"]),
        ("Ansible", ["ansible"]),
        ("Jenkins", ["jenkins"]),
        ("CI/CD", ["ci/cd", "continuous integration", "continuous deployment", "github actions", "gitlab ci"]),
        ("Serverless", ["serverless"]),
        ("Linux", ["linux", "unix", "ubuntu", "debian", "redhat", "centos"]),
        ("Cloudflare", ["cloudflare"]),
        ("Vercel", ["vercel"])
    ],
    "AI, ML & Data Science": [
        ("Machine Learning", ["machine learning", "ml"]),
        ("Deep Learning", ["deep learning", "dl"]),
        ("NLP", ["nlp", "natural language processing"]),
        ("Computer Vision", ["computer vision", "cv"]),
        ("Data Science", ["data science", "analytics"]),
        ("Large Language Models", ["llm", "large language models", "gpt", "llama", "claude", "openai"]),
        ("LangChain", ["langchain"]),
        ("RAG", ["rag", "retrieval augmented generation"]),
        ("Hadoop", ["hadoop"]),
        ("Spark", ["spark", "pyspark"]),
        ("Tableau", ["tableau"]),
        ("Power BI", ["power bi", "powerbi"])
    ],
    "Concepts & Methodologies": [
        ("System Design", ["system design", "architecture"]),
        ("Microservices", ["microservices", "soa"]),
        ("Agile", ["agile", "scrum", "kanban"]),
        ("DevOps", ["devops"]),
        ("REST API", ["rest api", "restful", "apis"]),
        ("GraphQL", ["graphql"]),
        ("gRPC", ["grpc"]),
        ("Git", ["git", "github", "gitlab", "bitbucket"]),
        ("TDD", ["tdd", "test driven development", "unit testing"]),
        ("OOP", ["oop", "object oriented programming", "solid principles"])
    ],
    "Soft Skills": [
        ("Leadership", ["leadership", "leading", "team lead"]),
        ("Communication", ["communication", "writing", "presentation"]),
        ("Teamwork", ["teamwork", "collaboration", "collaborative"]),
        ("Problem Solving", ["problem solving", "critical thinking"]),
        ("Time Management", ["time management", "organization"]),
        ("Adaptability", ["adaptability", "flexibility"]),
        ("Mentorship", ["mentorship", "mentoring", "coaching"]),
        ("Project Management", ["project management", "scrum master"])
    ]
}

def clean_text(text):
    """Normalize text by converting to lowercase and stripping excess whitespace."""
    if not text:
        return ""
    # Lowcase and clean spacing
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    """
    Scans clean text for skills based on SKILL_TAXONOMY patterns.
    Returns a dictionary mapping skill categories to lists of matched skill names.
    Also returns a set of all extracted skill names.
    """
    cleaned = clean_text(text)
    extracted_by_category = {}
    all_extracted_skills = set()
    
    for category, skills_list in SKILL_TAXONOMY.items():
        matched_skills = []
        for skill_name, patterns in skills_list:
            for pattern in patterns:
                # Compile regex with boundary checks
                # If pattern starts/ends with alphanumeric, we enforce word boundary
                # Special handling for patterns containing symbols (+, #, ., -)
                if any(char in pattern for char in ['+', '#', '.', '-']):
                    # Use custom boundaries that allow non-alphanumeric chars
                    regex_str = rf"(?:^|[^a-zA-Z0-9+#.-]){pattern}(?:$|[^a-zA-Z0-9+#.-])"
                else:
                    regex_str = rf"\b{pattern}\b"
                
                # Check for match
                if re.search(regex_str, cleaned):
                    matched_skills.append(skill_name)
                    all_extracted_skills.add(skill_name)
                    break # Matches one alias, move to next skill
                    
        if matched_skills:
            extracted_by_category[category] = matched_skills
            
    return extracted_by_category, all_extracted_skills

def calculate_jaccard_similarity(text1, text2):
    """Calculate basic keyword similarity based on unique word tokens, excluding stopwords."""
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    
    # Tokenize words using alphanumeric characters
    words1 = set(re.findall(r'\b[a-z]{2,}\b', clean_text(text1))) - stopwords
    words2 = set(re.findall(r'\b[a-z]{2,}\b', clean_text(text2))) - stopwords
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)
