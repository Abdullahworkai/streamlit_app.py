import streamlit as st
import PyPDF2
import json
import pandas as pd
import re
import os
import asyncio
import logging
from datetime import datetime
from io import BytesIO
import time

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crew.log'),
        logging.StreamHandler()
    ]
)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""

class PDFProcessor:
    """Handle PDF text extraction and parsing"""
    
    def __init__(self):
        self.cadet_responses = []
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting PDF text: {e}")
            return None
    
    def parse_pdf_content(self, pdf_text):
        """Parse PDF content to extract cadet responses"""
        if not pdf_text:
            return []
        
        # Split by page breaks and sections
        sections = re.split(r'Page \d+ of \d+|Essay Question|Test:', pdf_text)
        cadet_responses = []
        
        for section in sections:
            if not section.strip():
                continue
            
            # Accept both numbers AND letters as cadet identifiers
            cadet_match = re.search(r'For:\s*([A-Za-z0-9]+)', section)
            if not cadet_match:
                continue
            
            cadet_number = cadet_match.group(1)
            
            # Extract book number
            book_match = re.search(r'Book\s+(\d+)', section, re.IGNORECASE)
            book_number = f"Book_{book_match.group(1)}" if book_match else "Unknown"
            
            # Extract question
            question_match = re.search(r'Write a paragraph between[^.]*\.', section)
            question = question_match.group(0) if question_match else ""
            
            additional_question_match = re.search(r'-\s*Write about[^.]*\.', section)
            if additional_question_match:
                question += " " + additional_question_match.group(0)
            
            # Extract response
            response = ""
            question_end_pattern = r'-\s*Write about[^.]*\.'
            question_end_match = re.search(question_end_pattern, section)
            
            if question_end_match:
                after_question = section[section.find(question_end_match.group(0)) + len(question_end_match.group(0)):]
                response_match = re.search(r'^(.*?)(?:Instructor Comments:|$)', after_question, re.DOTALL)
                if response_match:
                    response = response_match.group(1).strip()
                    response = re.sub(r'\s+', ' ', response).strip()
            
            # Clean response
            response = self.clean_response(response)
            
            if cadet_number and question and response:
                cadet_responses.append({
                    'cadetNumber': cadet_number,
                    'bookNumber': book_number,
                    'question': question,
                    'response': response
                })
        
        logging.info(f"Extracted {len(cadet_responses)} cadet responses")
        return cadet_responses
    
    def clean_response(self, response):
        """Clean response text by removing unwanted sections"""
        response = re.sub(r'Section \d+', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s+', ' ', response).strip()
        return response

class KLPLoader:
    """Handle loading and management of Key Learning Points"""
    
    def __init__(self, klp_file_path):
        self.klp_file_path = klp_file_path
        self.klps = self.load_klps()
    
    def load_klps(self):
        """Load KLPs from JSON file"""
        try:
            with open(self.klp_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading KLPs: {e}")
            return {}
    
    def get_klp_for_book(self, book_number):
        """Get KLPs for a specific book number"""
        if book_number in self.klps:
            return self.klps[book_number]
        return {"Grammatical Structures": "No specific structures found", "Vocabulary Lists": "No vocabulary found"}
    
    def get_vocabulary_list(self, book_number):
        """Get ONLY vocabulary list for the book - direct injection"""
        klp = self.get_klp_for_book(book_number)
        return klp.get('Vocabulary Lists', 'No vocabulary found')
    
    def get_grammar_structures(self, book_number):
        """Get ONLY grammar structures for the book - direct injection"""
        klp = self.get_klp_for_book(book_number)
        return klp.get('Grammatical Structures', 'No grammar structures found')

class FeedbackManager:
    """Handle feedback and memory management"""
    
    def __init__(self, memory_file="long_term_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()
    
    def load_memory(self):
        """Load existing memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Error loading memory: {e}")
            return []
    
    def save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving memory: {e}")
    
    def add_feedback(self, input_text, output_result, rating, comment):
        """Add new feedback to memory"""
        feedback_entry = {
            "input": input_text[:500],
            "output": str(output_result)[:500],
            "feedback": {
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
        }
        self.memory.append(feedback_entry)
        self.save_memory()
        logging.info(f"Added feedback: rating={rating}, comment={comment[:50]}...")
    
    def get_latest_feedback(self):
        """Get the most recent feedback for adaptation"""
        if self.memory:
            latest = self.memory[-1]
            return f"Past feedback: Rating {latest['feedback']['rating']}/5 - {latest['feedback']['comment']}"
        return "No previous feedback available."

class CrewAIManager:
    """Manage CrewAI agents and evaluation process with kickoff_for_each_async"""
    
    def __init__(self, klp_loader, feedback_manager):
        self.klp_loader = klp_loader
        self.feedback_manager = feedback_manager
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=3000
        )
        
        # Initialize agents ONCE - reused for all students
        self.vocabulary_agent = self.create_vocabulary_agent()
        self.grammar_agent = self.create_grammar_agent()
        self.task_achievement_agent = self.create_task_achievement_agent()
        self.range_accuracy_agent = self.create_range_accuracy_agent()
        self.quality_agent = self.create_quality_agent()
        self.manager_agent = self.create_manager_agent()
    
    def create_manager_agent(self):
        """Create manager agent for hierarchical delegation"""
        return Agent(
            role="Senior Evaluation Manager",
            goal="""Coordinate comprehensive student writing evaluations by delegating 
            to specialized agents and ensuring consistent standards across all assessments.""",
            backstory="""You are an experienced academic evaluation coordinator with 20+ years 
            in language assessment. You oversee a team of specialized evaluators and ensure 
            every student receives fair, thorough, and constructive feedback. You delegate 
            vocabulary tasks to the Vocabulary Evaluator, grammar tasks to the Grammar Evaluator, 
            task achievement to the Task Achievement Specialist, range and accuracy to the 
            Range and Accuracy Specialist, and final synthesis to the Quality Assurance Lead.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            memory=False,
            max_iter=15,
        )
    
    def create_vocabulary_agent(self):
        """Create vocabulary evaluation agent with EXPLICIT scoring criteria"""
        return Agent(
            role="Vocabulary Evaluator",
            goal="Evaluate vocabulary usage strictly based on the provided KLP vocabulary list",
            backstory="""You are a vocabulary assessment specialist. You ONLY count vocabulary 
            words that appear in the provided KLP vocabulary list. You follow the exact scoring 
            rubric without deviation. You always return valid JSON.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=False,
            max_iter=10,
        )
    
    def create_grammar_agent(self):
        """Create grammar evaluation agent with EXPLICIT scoring criteria"""
        return Agent(
            role="Grammar Evaluator",
            goal="Evaluate grammar structure usage strictly based on the provided KLP grammar structures",
            backstory="""You are a grammar assessment specialist. You ONLY count grammar 
            structures that appear in the provided KLP grammar list. You follow the exact 
            scoring rubric without deviation. You always return valid JSON.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=False,
            max_iter=10,
        )
    
    def create_task_achievement_agent(self):
        """Create task achievement evaluation agent with EXPLICIT scoring criteria"""
        return Agent(
            role="Task Achievement Specialist", 
            goal="Evaluate how completely students answer all parts and stay within word limit",
            backstory="""You are an expert in evaluating task completion. You assess whether 
            students answered all parts of the question and stayed within the word limit. 
            You follow the exact scoring rubric without deviation. You always return valid JSON.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=False,
            max_iter=10,
        )
    
    def create_range_accuracy_agent(self):
        """Create range and accuracy evaluation agent with EXPLICIT scoring criteria"""
        return Agent(
            role="Range and Accuracy Specialist",
            goal="Evaluate grammar, punctuation, and spelling quality",
            backstory="""You are a language quality specialist who evaluates the overall 
            quality of grammar, punctuation, and spelling. You assess the range of structures 
            and accuracy of language use. You follow the exact scoring rubric without deviation. 
            You always return valid JSON.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=False,
            max_iter=10,
        )
    
    def create_quality_agent(self):
        """Create quality assurance agent"""
        return Agent(
            role="Quality Assurance Lead",
            goal="Synthesize all evaluations into comprehensive, actionable feedback",
            backstory="""You are the final quality check for all student assessments. 
            You review scores from all specialists, ensure consistency, and craft final 
            feedback that is constructive, specific, and helpful for student improvement. 
            You always return valid JSON.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=False,
            max_iter=10,
        )
    
    def parse_agent_output(self, output_text):
        """Parse JSON from agent output with improved extraction"""
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*|\s*```', '', str(output_text))
            
            # Try to find JSON in the output
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                logging.info(f"Successfully parsed JSON: {parsed}")
                return parsed
            
            # If no JSON found, try to extract scores from text
            logging.warning("No JSON found, attempting text extraction")
            scores = {}
            
            vocab_match = re.search(r'vocabulary[_\s]+score[:\s]+(\d+)', str(output_text), re.IGNORECASE)
            if vocab_match:
                scores['vocabulary_score'] = int(vocab_match.group(1))
            
            grammar_match = re.search(r'grammar[_\s]+score[:\s]+(\d+)', str(output_text), re.IGNORECASE)
            if grammar_match:
                scores['grammar_score'] = int(grammar_match.group(1))
            
            task_match = re.search(r'task[_\s]+achievement[_\s]+score[:\s]+(\d+)', str(output_text), re.IGNORECASE)
            if task_match:
                scores['task_achievement_score'] = int(task_match.group(1))
            
            range_match = re.search(r'range[_\s]+(?:and[_\s]+)?accuracy[_\s]+score[:\s]+(\d+)', str(output_text), re.IGNORECASE)
            if range_match:
                scores['range_accuracy_score'] = int(range_match.group(1))
            
            feedback_match = re.search(r'(?:final[_\s]+)?feedback[:\s]+"?([^"\n]+)"?', str(output_text), re.IGNORECASE)
            if feedback_match:
                scores['final_feedback'] = feedback_match.group(1).strip()
            
            if scores:
                logging.info(f"Extracted scores from text: {scores}")
                return scores
            
            logging.error("No scores found in output")
            return None
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logging.error(f"Error parsing output: {e}")
            return None
    
    def validate_scores(self, result_data):
        """Validate and clamp scores to correct ranges"""
        # Vocabulary: 0-10
        if 'vocabulary_score' in result_data:
            result_data['vocabulary_score'] = max(0, min(10, result_data['vocabulary_score']))
        
        # Grammar: 0-10
        if 'grammar_score' in result_data:
            result_data['grammar_score'] = max(0, min(10, result_data['grammar_score']))
        
        # Task Achievement: 0-40
        if 'task_achievement_score' in result_data:
            result_data['task_achievement_score'] = max(0, min(40, result_data['task_achievement_score']))
        
        # Range & Accuracy: 0-40
        if 'range_accuracy_score' in result_data:
            result_data['range_accuracy_score'] = max(0, min(40, result_data['range_accuracy_score']))
        
        return result_data
    
    async def evaluate_students(self, cadet_responses):
        """Run parallel evaluation using kickoff_for_each - OPTIMAL METHOD!"""
        if not cadet_responses:
            return []
        
        total = len(cadet_responses)
        logging.info(f"Starting evaluation of {total} students using kickoff_for_each")
        
        # Get book number (assuming all students are from same book)
        book_number = cadet_responses[0]['bookNumber']
        
        # DIRECT KLP INJECTION - No hallucination risk
        vocabulary_list = self.klp_loader.get_vocabulary_list(book_number)
        grammar_structures = self.klp_loader.get_grammar_structures(book_number)
        
        logging.info(f"All students from {book_number}")
        logging.info(f"Vocabulary list length: {len(str(vocabulary_list))}")
        logging.info(f"Grammar structures length: {len(str(grammar_structures))}")
        
        # Define tasks with placeholders for student data
        vocabulary_task = Task(
            description="""
STUDENT RESPONSE:
{response}

KLP VOCABULARY LIST FOR THIS BOOK (ONLY count words from this list):
""" + vocabulary_list + """

SCORING CRITERIA - Vocabulary Usage (0-10 points):
- 0 = No response
- 2 = Uses one vocabulary word but incorrectly
- 4 = Attempts one or two vocabulary words with limited success (errors or partial usage)
- 6 = Uses one or two vocabulary words but not fully accurate across response
- 8 = Uses three vocabulary words with minor issues
- 10 = Uses three vocabulary words correctly & appropriately

IMPORTANT: Be extremely strict - ONLY count vocabulary words that appear in the KLP vocabulary list above.

Count the vocabulary words from the KLP list that appear in the student's response. Check if they are used correctly and appropriately.

You MUST return ONLY valid JSON in this exact format:
{{"vocabulary_score": <number 0-10>, "vocab_words_found": ["word1", "word2"]}}
            """,
            expected_output='Valid JSON: {"vocabulary_score": X, "vocab_words_found": ["word1", "word2"]}'
        )
        
        grammar_task = Task(
            description="""
STUDENT RESPONSE:
{response}

KLP GRAMMAR STRUCTURES FOR THIS BOOK (ONLY count structures from this list):
""" + grammar_structures + """

SCORING CRITERIA - Grammar Usage (0-10 points):
- 0 = No response
- 2 = Uses one grammar point but incorrectly
- 4 = Attempts one grammar point with limited success (errors or partial usage)
- 6 = Uses one grammar point but not fully accurate across response
- 8 = Uses one grammar point with minor issues
- 10 = Uses one grammar point correctly & appropriately

IMPORTANT: Be extremely strict - ONLY count grammar structures that appear in the KLP grammar list above.

Identify which grammar structure(s) from the KLP list appear in the student's response. Check if they are used correctly.

You MUST return ONLY valid JSON in this exact format:
{{"grammar_score": <number 0-10>, "grammar_structures_found": ["structure1"]}}
            """,
            expected_output='Valid JSON: {"grammar_score": X, "grammar_structures_found": ["structure1"]}'
        )
        
        task_achievement_task = Task(
            description="""
QUESTION:
{question}

STUDENT RESPONSE:
{response}

SCORING CRITERIA - Task Achievement (0-40 points):
- 0 = No response
- 8 = Minimal, very few words, barely addresses question
- 16 = Half response, covers about half with major gaps
- 24 = Partial, answers most parts, some gaps or short on words
- 32 = Good, mostly developed, minor improvements needed
- 40 = Full, complete, well-developed, within limit

Evaluate:
1. Did the student answer ALL parts of the question?
2. Is the response within the word limit?
3. How well-developed is the response?

You MUST return ONLY valid JSON in this exact format:
{{"task_achievement_score": <number 0-40>, "feedback": "one sentence explanation"}}
            """,
            expected_output='Valid JSON: {"task_achievement_score": X, "feedback": "one sentence"}'
        )
        
        range_accuracy_task = Task(
            description="""
STUDENT RESPONSE:
{response}

SCORING CRITERIA - Range & Accuracy (0-40 points):
Grammar, punctuation, and spelling quality:
- 0 = No response
- 8 = Minimal response. Numerous errors in structure, punctuation, spelling (hard to understand)
- 16 = Half response. Some correct structures but frequent errors in spelling, punctuation, sentence formation
- 24 = Partial response. Reasonable range of structures and vocabulary but noticeable errors
- 32 = Good response. Good range of vocabulary and grammatical structures with minor errors
- 40 = Full response. Excellent structure, punctuation, and accuracy with very few minor errors

Evaluate:
1. Range of grammatical structures used
2. Range of vocabulary used
3. Spelling accuracy
4. Punctuation accuracy
5. Overall comprehensibility

You MUST return ONLY valid JSON in this exact format:
{{"range_accuracy_score": <number 0-40>, "feedback": "one sentence explanation"}}
            """,
            expected_output='Valid JSON: {"range_accuracy_score": X, "feedback": "one sentence"}'
        )
        
        quality_task = Task(
            description="""
CADET: {cadetNumber}
BOOK: {bookNumber}
RESPONSE: {response}

Review all specialist evaluations and provide:
1. A comprehensive final feedback summary (2-3 sentences)
2. Specific strengths the student demonstrated
3. Specific areas for improvement

You MUST return ONLY valid JSON in this exact format:
{{"final_feedback": "comprehensive 2-3 sentence assessment"}}
            """,
            expected_output='Valid JSON: {"final_feedback": "comprehensive assessment"}'
        )
        
        # Create ONE crew that will be reused for all students
        crew = Crew(
            agents=[self.vocabulary_agent, self.grammar_agent,
                   self.task_achievement_agent, self.range_accuracy_agent, self.quality_agent],
            tasks=[vocabulary_task, grammar_task, task_achievement_task,
                  range_accuracy_task, quality_task],
            process=Process.hierarchical,
            manager_agent=self.manager_agent,
            verbose=True
        )
        
        # Prepare inputs for all students
        inputs = []
        for response in cadet_responses:
            inputs.append({
                'cadetNumber': response['cadetNumber'],
                'bookNumber': response['bookNumber'],
                'question': response['question'],
                'response': response['response']
            })
        
        try:
            # Run kickoff_for_each - parallel processing by CrewAI
            logging.info(f"Calling kickoff_for_each with {len(inputs)} students")
            
            crew_results = await asyncio.to_thread(
                crew.kickoff_for_each,
                inputs=inputs
            )
            
            logging.info(f"Received {len(crew_results)} results from crew")
            
            # Process results
            final_results = []
            for idx, crew_result in enumerate(crew_results):
                try:
                    # Initialize result data
                    result_data = {
                        'cadet_id': inputs[idx]['cadetNumber'],
                        'book_number': inputs[idx]['bookNumber'],
                        'vocab_score': 0,
                        'grammar_score': 0,
                        'task_achievement_score': 0,
                        'range_accuracy_score': 0,
                        'final_feedback': "Evaluation incomplete"
                    }
                    
                    # Parse result
                    if hasattr(crew_result, 'raw'):
                        parsed = self.parse_agent_output(crew_result.raw)
                        if parsed:
                            result_data.update(parsed)
                    
                    # Fallback: try tasks_output
                    if result_data['vocab_score'] == 0 and hasattr(crew_result, 'tasks_output'):
                        for task_output in crew_result.tasks_output:
                            parsed = self.parse_agent_output(task_output.raw)
                            if parsed:
                                result_data.update(parsed)
                    
                    # Validate scores
                    result_data = self.validate_scores(result_data)
                    final_results.append(result_data)
                    
                    logging.info(f"Processed student {idx+1}/{total}: {result_data}")
                    
                except Exception as e:
                    logging.error(f"Error processing result {idx}: {e}", exc_info=True)
                    final_results.append({
                        'cadet_id': inputs[idx]['cadetNumber'],
                        'book_number': inputs[idx]['bookNumber'],
                        'vocab_score': 0,
                        'grammar_score': 0,
                        'task_achievement_score': 0,
                        'range_accuracy_score': 0,
                        'final_feedback': f"Processing error: {str(e)[:100]}"
                    })
            
            logging.info(f"Completed evaluation: {len(final_results)}/{total} successful")
            return final_results
            
        except Exception as e:
            logging.error(f"Error in kickoff_for_each: {e}", exc_info=True)
            
            # Return error results for all students
            return [{
                'cadet_id': response['cadetNumber'],
                'book_number': response['bookNumber'],
                'vocab_score': 0,
                'grammar_score': 0,
                'task_achievement_score': 0,
                'range_accuracy_score': 0,
                'final_feedback': f"Batch evaluation error: {str(e)[:100]}"
            } for response in cadet_responses]

def create_excel_download(results):
    """Create Excel file for download"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'cadet_id', 'book_number', 'vocab_score', 'grammar_score',
        'task_achievement_score', 'range_accuracy_score', 'final_feedback'
    ]
    
    if all(col in df.columns for col in column_order):
        df = df[column_order]
    
    # Create Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Evaluation Results', index=False)
    
    output.seek(0)
    return output.getvalue()

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Student Writing Evaluator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Student Writing PDF Evaluator")
    st.markdown("Upload student writing PDFs for AI-powered evaluation using CrewAI")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'klp_loader' not in st.session_state:
        st.session_state.klp_loader = None
    if 'feedback_manager' not in st.session_state:
        st.session_state.feedback_manager = FeedbackManager()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.error("❌ OpenAI API Key: Missing")
            st.info("Add OPENAI_API_KEY to secrets or environment variables")
        
        # KLP file upload
        st.subheader("📚 Key Learning Points (KLPs)")
        klp_file = st.file_uploader(
            "Upload KLPs JSON file",
            type=['json'],
            help="Upload the book_vocabulary_grammar_lists.json file"
        )
        
        if klp_file:
            # Save uploaded file temporarily
            with open("temp_klps.json", "wb") as f:
                f.write(klp_file.getbuffer())
            st.session_state.klp_loader = KLPLoader("temp_klps.json")
            st.success("✅ KLPs loaded successfully!")
        
        # Memory management
        st.subheader("🧠 Memory Management")
        if st.button("Clear Memory"):
            st.session_state.feedback_manager.memory = []
            st.session_state.feedback_manager.save_memory()
            st.success("Memory cleared!")
        
        # Info section
        st.markdown("---")
        st.info("**Note:** Cadet IDs accept both numbers (1234) and letters (ABCD) as identifiers.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 PDF Upload and Processing")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a student writing PDF for evaluation"
        )
        
        if uploaded_file and st.session_state.klp_loader:
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        processor = PDFProcessor()
                        pdf_text = processor.extract_text_from_pdf(uploaded_file)
                        
                        if pdf_text:
                            cadet_responses = processor.parse_pdf_content(pdf_text)
                            
                            if cadet_responses:
                                st.success(f"✅ Extracted {len(cadet_responses)} student responses")
                                
                                # Display extracted data
                                with st.expander("👀 Extracted Data Preview"):
                                    for i, response in enumerate(cadet_responses[:3]):
                                        st.write(f"**Cadet {response['cadetNumber']} ({response['bookNumber']})**")
                                        st.write(f"Question: {response['question'][:100]}...")
                                        st.write(f"Response: {response['response'][:200]}...")
                                        st.write("---")
                                
                                # Run evaluation
                                with st.spinner("🤖 Running AI evaluation..."):
                                    crew_manager = CrewAIManager(
                                        st.session_state.klp_loader,
                                        st.session_state.feedback_manager
                                    )
                                    
                                    results = asyncio.run(crew_manager.evaluate_students(cadet_responses))
                                    st.session_state.results = results
                                
                                st.success(f"✅ Evaluation complete for {len(results)} students!")
                                
                            else:
                                st.error("❌ No student responses found in PDF")
                        else:
                            st.error("❌ Failed to extract text from PDF")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        logging.error(f"PDF processing error: {e}")
        
        elif uploaded_file and not st.session_state.klp_loader:
            st.warning("⚠️ Please upload the KLPs JSON file first (in sidebar)")
        
        # Display results
        if st.session_state.results:
            st.header("📊 Evaluation Results")
            
            # Results table
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df, use_container_width=True)
            
            # Download Excel
            excel_data = create_excel_download(st.session_state.results)
            if excel_data:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col2:
        st.header("💬 Feedback System")
        
        if st.session_state.results:
            st.subheader("Rate the Evaluation")
            
            # Rating slider
            rating = st.slider(
                "How accurate was the evaluation?",
                min_value=1,
                max_value=5,
                value=3,
                help="Rate the quality of the AI evaluation (1=Poor, 5=Excellent)"
            )
            
            # Comment box
            comment = st.text_area(
                "Additional Feedback",
                placeholder="Provide specific feedback to improve future evaluations...",
                help="Your feedback will be used to adapt the AI evaluation for future runs"
            )
            
            if st.button("Submit Feedback"):
                if comment.strip():
                    # Get sample input/output for memory
                    sample_input = st.session_state.results[0]['final_feedback'] if st.session_state.results else ""
                    sample_output = str(st.session_state.results)
                    
                    st.session_state.feedback_manager.add_feedback(
                        sample_input, sample_output, rating, comment
                    )
                    
                    st.success("✅ Feedback saved! It will be used to improve future evaluations.")
                else:
                    st.warning("⚠️ Please provide a comment with your feedback.")
        
        # Memory status
        st.subheader("📈 Memory Status")
        memory_count = len(st.session_state.feedback_manager.memory)
        st.metric("Feedback Entries", memory_count)
        
        if memory_count > 0:
            latest_feedback = st.session_state.feedback_manager.memory[-1]
            st.write("**Latest Feedback:**")
            st.write(f"⭐ Rating: {latest_feedback['feedback']['rating']}/5")
            st.write(f"💭 Comment: {latest_feedback['feedback']['comment'][:100]}...")

if __name__ == "__main__":
    main()
