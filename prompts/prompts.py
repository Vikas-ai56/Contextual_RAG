SEARCH_QUERY_PROMPT="""
You are an expert insurance claims analyst. Your task is to generate search queries to efficiently
retrieve data from the database
        
These are the set of reasoning questions and there response respectively given by the user
<reasoning_sub_questions>
{reason_queries}
</reasoning_sub_questions>

the response
<resoning_response>
{user_response_to_rqs}
</resoning_response>
        
Generate search queries (maximum of 5) based on the information obtained by the user.

--- EXAMPLE 1 ---
"reasoning_sub_questions" : [What was the exact name of the medical procedure you had?,
Did your doctor get approval from the our company *before* surgery]

"reasoning_responses" : I had a planned ACL reconstruction on my knee.
My doctor's office said they handled the approval beforehand.

search_queries : ["Coverage details for elective orthopedic surgeries like ACL reconstruction.",
"Pre-authorization requirements and process for planned surgery.",
"Waiting period for non-emergency knee ligament surgery.",
"Exclusions related to sports injuries or existing orthopedic condition."]

--- EXAMPLE 2 ---
"reasoning_sub_questions" : [To check the policy rules, is your liver condition related to alcohol use?,
Name of the hospital or clinic where you took treatment?]

"reasoning_responses" : My diagnosis was for non-alcoholic fatty liver disease (NAFLD). I do not consume alcohol.
At Mercy General Hospital.

search_queries : ["Policy coverage for metabolic conditions such as non-alcoholic fatty liver disease.",
"Exclusionary clauses for liver treatments not related to alcohol or substance abuse.",
"List of in-network hospitals for major medical treatments.",
"Coverage limits for inpatient treatment at Mercy General Hospital."]
"""


PLANNER_PROMPT = """
You are an expert insurance claims analyst. Your task is to deconstruct a user's query 
into a logical plan for investigation.

Generate "reason_queries". These are clarifying questions a human analyst would 
need to ask the user to gather all necessary information. They should probe for potential 
policy exclusions, waiting periods, and network status.

Generate "key_concepts". These are the key extracts/keywords present in the question which
is related to the policy

You are an expert insurance claims analyst. Your task is to deconstruct a user's query 
into a logical plan for investigation.

Generate "Reasoning Queries". These are clarifying questions a human analyst would 
need to ask the user to gather all necessary information. They should probe for potential 
policy exclusions, waiting periods, and network status.

IMPORTANT:- the queries you generate **MUST** be in very simple english and also make sure you do not ask more than 5-6 reasoning query

Example-1
"user_query": "I want to reimburse my insurance amount for liver treatment."

"key_concepts":[1. "Liver treatment",
2. "Non-alcoholic fatty liver disease (NAFLD)"]
"reason_queries": [1. What was the exact name of the medical procedure you had?,
2. Did your doctor get approval from the our company *before* surgery?]

Example-2
"user_query": "I am pregnant and want to know about coverage for delivery."

"key_concepts": [1. Pre-pregnancy,
2. Coverage for child delivery] 
"reason_queries": [Which type of Insurance poilcy you have?,
Is this a normal pregnancy or are there any problems?,
Do you plan to use a hospital that is part of the insurance network?]
"""

# Example implementation
PLANNER_PROMPT = PLANNER_PROMPT.format(format_instructions="my_format_instructions")

GENERATOR_PROMPT = """
You are a meticulous insurance claims assistant. Your task is to provide a final, 
comprehensive answer to the user's query based ONLY on the evidence provided 
from the policy document. Do not invent information or make assumptions.

Use the 'Reasoning Questions' as a guide to structure your analysis of the evidence. 
Address each point if possible and cite the information clearly.

**User's Original Query:**

<user_query>
{user_query}
</user_query>

**Evidence Corpus from policy document**
        
<evidence>
{final_results}
</evidence>

Provide a clear, structured, and helpful final answer.
"""


DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (maximum of 50 words) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and NOTHING ELSE.
"""

FEW_SHOT_EXAMPLES = [
    {
        "user_query": "I want to reimburse my insurance amount for liver treatment.",
        "reasoning_queries": """- Is this treatment related to alcohol consumption?
- What specific medical procedures were performed?
- Was the treatment received at an in-network or out-of-network hospital?""",
        "search_queries": """- policy exclusions for alcohol-related liver conditions
- coverage details for liver surgery and related treatments
- reimbursement rules for in-network vs out-of-network hospitals
- waiting period for organ-related critical illnesses
- documents required for submitting a major medical claim"""
    },
    {
        "user_query": "I am pregnant and want to know about coverage for delivery.",
        "reasoning_queries": """- How long have you had this insurance policy?
- Is this a routine pregnancy or are there complications?
- Do you plan to use a hospital that is part of the insurance network?""",
        "search_queries": """- maternity and childbirth benefits coverage
- waiting period for pregnancy-related claims
- coverage limits for normal delivery vs. caesarean section
- in-network hospitals and clinics for maternity care
- coverage for newborn baby care and post-natal checkups"""
    },
    {
        "user_query": "I had a bike accident and broke my arm. What do I do?",
        "reasoning_queries": """- Was the treatment performed in an emergency room?
- Does the policy have any exclusions for injuries from hazardous activities or sports?
- Will follow-up care like physiotherapy be required?""",
        "search_queries": """- accidental injury and emergency medical coverage
- policy exclusions related to adventure sports or hazardous activities
- coverage for post-accident rehabilitation and physiotherapy
- procedure for filing an accident insurance claim
- list of documents required for accident reimbursement"""
    }
]