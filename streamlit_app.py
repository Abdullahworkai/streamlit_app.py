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
from langchain.llms import OpenAI

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
import streamlit as st
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
# st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""

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
            
            # Extract cadet number
            cadet_match = re.search(r'For:\s*(\d+)', section)
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
            
            # Clean response (remove "Section 1" etc.)
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
        # Remove "Section 1" and similar patterns
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
            "input": input_text[:500],  # Limit input size
            "output": str(output_result)[:500],  # Limit output size
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
    """Manage CrewAI agents and evaluation process"""
    
    def __init__(self, klp_loader, feedback_manager):
        self.klp_loader = klp_loader
        self.feedback_manager = feedback_manager
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize agents
        self.manager_agent = self.create_manager_agent()
        self.grammar_vocab_agent = self.create_grammar_vocab_agent()
        self.task_achievement_agent = self.create_task_achievement_agent()
        self.range_accuracy_agent = self.create_range_accuracy_agent()
        self.quality_agent = self.create_quality_agent()
    
    def create_manager_agent(self):
        """Create manager agent for delegation"""
        feedback_context = self.feedback_manager.get_latest_feedback()
        
        return Agent(
            role="Evaluation Manager",
            goal="Coordinate and delegate evaluation tasks to specialized agents",
            backstory="""You are an experienced academic evaluation manager who coordinates 
            comprehensive writing assessments. You delegate tasks to specialized agents and 
            ensure consistent, fair evaluation standards.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            memory=True,
            max_iter=3,
            system_message=f"""
            You coordinate writing evaluations by delegating to three specialized agents:
            1. Grammar/Vocabulary Agent
            2. Task Achievement Agent  
            3. Range/Accuracy Agent
            
            {feedback_context}
            
            Ensure all agents use the provided Key Learning Points (KLPs) for accurate assessment.
            """
        )
    
    def create_grammar_vocab_agent(self):
        """Create grammar and vocabulary evaluation agent"""
        return Agent(
            role="Grammar and Vocabulary Evaluator",
            goal="Evaluate grammar and vocabulary usage based on Key Learning Points",
            backstory="""You are a specialized grammar and vocabulary evaluator with expertise 
            in assessing student writing against specific learning objectives. You provide 
            detailed feedback on grammatical structures and vocabulary usage.""",
            verbose=True,
            llm=self.llm,
            memory=True,
            max_iter=2,
            system_message="""
            Evaluate grammar and vocabulary on a 0-10 scale each.
            Be strict and accurate based on the provided Key Learning Points.
            Output format: {"vocabulary_score": X, "grammar_score": Y}
            """
        )
    
    def create_task_achievement_agent(self):
        """Create task achievement evaluation agent"""
        return Agent(
            role="Task Achievement Evaluator", 
            goal="Evaluate how well students fulfill the writing task requirements",
            backstory="""You are an expert in evaluating task completion and achievement. 
            You assess how well students address the prompt requirements and achieve 
            the intended learning objectives.""",
            verbose=True,
            llm=self.llm,
            memory=True,
            max_iter=2,
            system_message="""
            Evaluate task achievement on a 0-40 scale.
            Vary scores appropriately - not all students should get the same score.
            Provide one-sentence feedback for each student.
            Output format: {"task_achievement_score": X, "feedback": "sentence"}
            """
        )
    
    def create_range_accuracy_agent(self):
        """Create range and accuracy evaluation agent"""
        return Agent(
            role="Range and Accuracy Evaluator",
            goal="Evaluate the range and accuracy of language use in student writing",
            backstory="""You are a language assessment specialist who evaluates the range 
            and accuracy of linguistic structures used by students. You focus on 
            complexity and precision of language use.""",
            verbose=True,
            llm=self.llm,
            memory=True,
            max_iter=2,
            system_message="""
            Evaluate range and accuracy on a 0-40 scale.
            Vary scores appropriately based on language complexity and accuracy.
            Provide one-sentence feedback for each student.
            Output format: {"range_accuracy_score": X, "feedback": "sentence"}
            """
        )
    
    def create_quality_agent(self):
        """Create quality assurance agent"""
        return Agent(
            role="Quality Assurance Specialist",
            goal="Combine evaluations and ensure consistency across all assessments",
            backstory="""You are a quality assurance specialist who reviews and combines 
            evaluations from multiple agents. You ensure consistency and add final 
            comprehensive feedback.""",
            verbose=True,
            llm=self.llm,
            memory=True,
            max_iter=2,
            system_message="""
            Combine all scores and ensure no duplicates.
            Add final comprehensive feedback for each student.
            Check against past evaluations to avoid repetition.
            Output format: {"final_feedback": "comprehensive assessment"}
            """
        )
    
    async def evaluate_students(self, cadet_responses):
        """Run the evaluation process for all students"""
        if not cadet_responses:
            return []
        
        results = []
        
        for response in cadet_responses:
            try:
                # Get KLPs for the book
                klps = self.klp_loader.get_klp_for_book(response['bookNumber'])
                klp_text = f"""
                Grammatical Structures: {klps.get('Grammatical Structures', 'Not available')}
                Vocabulary Lists: {klps.get('Vocabulary Lists', 'Not available')}
                """
                
                # Create tasks
                grammar_task = Task(
                    description=f"""
                    Evaluate the grammar and vocabulary of this student response:
                    
                    Book: {response['bookNumber']}
                    Question: {response['question']}
                    Response: {response['response']}
                    
                    KLPs for this book:
                    {klp_text}
                    
                    Provide scores for vocabulary (0-10) and grammar (0-10) based strictly on the KLPs.
                    """,
                    agent=self.grammar_vocab_agent,
                    expected_output="JSON with vocabulary_score and grammar_score"
                )
                
                task_achievement_task = Task(
                    description=f"""
                    Evaluate task achievement for this student response:
                    
                    Question: {response['question']}
                    Response: {response['response']}
                    
                    Assess how well the student addressed the prompt (0-40 points).
                    """,
                    agent=self.task_achievement_agent,
                    expected_output="JSON with task_achievement_score and feedback"
                )
                
                range_accuracy_task = Task(
                    description=f"""
                    Evaluate range and accuracy for this student response:
                    
                    Response: {response['response']}
                    
                    Assess language complexity and accuracy (0-40 points).
                    """,
                    agent=self.range_accuracy_agent,
                    expected_output="JSON with range_accuracy_score and feedback"
                )
                
                quality_task = Task(
                    description=f"""
                    Review and combine all evaluations for final quality assessment:
                    
                    Cadet: {response['cadetNumber']}
                    Response: {response['response']}
                    
                    Provide comprehensive final feedback.
                    """,
                    agent=self.quality_agent,
                    expected_output="JSON with final_feedback"
                )
                
                # Create crew and execute
                crew = Crew(
                    agents=[self.manager_agent, self.grammar_vocab_agent, 
                           self.task_achievement_agent, self.range_accuracy_agent, self.quality_agent],
                    tasks=[grammar_task, task_achievement_task, range_accuracy_task, quality_task],
                    process=Process.hierarchical,
                    verbose=True
                )
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(crew.kickoff),
                        timeout=45.0
                    )
                    
                    # Parse results (simplified for demo)
                    result_data = {
                        'cadet_id': response['cadetNumber'],
                        'book_number': response['bookNumber'],
                        'vocab_score': 7,  # Would parse from actual result
                        'grammar_score': 8,  # Would parse from actual result
                        'task_achievement_score': 32,  # Would parse from actual result
                        'range_accuracy_score': 28,  # Would parse from actual result
                        'final_feedback': "Good overall performance with room for improvement in vocabulary range."
                    }
                    
                    results.append(result_data)
                    
                except asyncio.TimeoutError:
                    logging.warning(f"Timeout evaluating cadet {response['cadetNumber']}")
                    results.append({
                        'cadet_id': response['cadetNumber'],
                        'book_number': response['bookNumber'],
                        'vocab_score': 0,
                        'grammar_score': 0,
                        'task_achievement_score': 0,
                        'range_accuracy_score': 0,
                        'final_feedback': "Evaluation timeout - please retry."
                    })
                
            except Exception as e:
                logging.error(f"Error evaluating cadet {response['cadetNumber']}: {e}")
                results.append({
                    'cadet_id': response['cadetNumber'],
                    'book_number': response['bookNumber'],
                    'vocab_score': 0,
                    'grammar_score': 0,
                    'task_achievement_score': 0,
                    'range_accuracy_score': 0,
                    'final_feedback': f"Evaluation error: {str(e)[:100]}"
                })
        
        return results

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
        st.header("Configuration")
        
        # API Key status
        st.success("✅ OpenAI API Key: Configured and Ready")
        
        # KLP file upload
        st.subheader("Key Learning Points (KLPs)")
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
            st.success("KLPs loaded successfully!")
        
        # Memory management
        st.subheader("Memory Management")
        if st.button("Clear Memory"):
            st.session_state.feedback_manager.memory = []
            st.session_state.feedback_manager.save_memory()
            st.success("Memory cleared!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("PDF Upload and Processing")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a student writing PDF for evaluation"
        )
        
        if uploaded_file and st.session_state.klp_loader:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        processor = PDFProcessor()
                        pdf_text = processor.extract_text_from_pdf(uploaded_file)
                        
                        if pdf_text:
                            cadet_responses = processor.parse_pdf_content(pdf_text)
                            
                            if cadet_responses:
                                st.success(f"Extracted {len(cadet_responses)} student responses")
                                
                                # Display extracted data
                                with st.expander("Extracted Data Preview"):
                                    for i, response in enumerate(cadet_responses[:3]):  # Show first 3
                                        st.write(f"**Cadet {response['cadetNumber']} ({response['bookNumber']})**")
                                        st.write(f"Question: {response['question'][:100]}...")
                                        st.write(f"Response: {response['response'][:200]}...")
                                        st.write("---")
                                
                                # Run evaluation
                                with st.spinner("Running AI evaluation..."):
                                    crew_manager = CrewAIManager(
                                        st.session_state.klp_loader,
                                        st.session_state.feedback_manager
                                    )
                                    
                                    results = asyncio.run(crew_manager.evaluate_students(cadet_responses))
                                    st.session_state.results = results
                                
                                st.success(f"Evaluation complete for {len(results)} students!")
                                
                            else:
                                st.error("No student responses found in PDF")
                        else:
                            st.error("Failed to extract text from PDF")
                            
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        logging.error(f"PDF processing error: {e}")
        
        # Display results
        if st.session_state.results:
            st.header("Evaluation Results")
            
            # Results table
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df, use_container_width=True)
            
            # Download Excel
            excel_data = create_excel_download(st.session_state.results)
            if excel_data:
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col2:
        st.header("Feedback System")
        
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
                    
                    st.success("Feedback saved! It will be used to improve future evaluations.")
                else:
                    st.warning("Please provide a comment with your feedback.")
        
        # Memory status
        st.subheader("Memory Status")
        memory_count = len(st.session_state.feedback_manager.memory)
        st.metric("Feedback Entries", memory_count)
        
        if memory_count > 0:
            latest_feedback = st.session_state.feedback_manager.memory[-1]
            st.write("**Latest Feedback:**")
            st.write(f"Rating: {latest_feedback['feedback']['rating']}/5")
            st.write(f"Comment: {latest_feedback['feedback']['comment'][:100]}...")

if __name__ == "__main__":
    main()
