"""Main Streamlit application"""
import streamlit as st
from utils.vector_store import initialize_pinecone, store_chunks, search_similar_chunks, get_index_stats
from utils.document_processor import process_document
from utils.chunking import chunk_text
from utils.llm import generate_completion, create_rag_prompt

def main():
    st.set_page_config(page_title="RAG DOC READER Q&A", layout="wide")
    st.title("📄 RAG DOC READER ")
    
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    total_chunks = 0
                    
                    for file in uploaded_files:
                        # Extract text
                        text = process_document(file)
                        if not text:
                            st.warning(f"Failed to process {file.name}")
                            continue
                        
                        # Chunk text
                        chunks = chunk_text(text)
                        st.info(f"{file.name}: {len(chunks)} chunks created")
                        
                        # Store in Pinecone
                        stored = store_chunks(index, chunks, file.name)
                        total_chunks += stored
                    
                    st.success(f"✅ Processed {total_chunks} chunks from {len(uploaded_files)} documents")
            else:
                st.warning("Please upload documents first")
    
    # Main area for querying
    st.header("💬 Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching and generating answer..."):
                # Search for similar chunks
                results = search_similar_chunks(index, query)
                
                if results['matches']:
                    # Extract context
                    context = "\n\n".join([match['metadata']['text'] for match in results['matches']])

                    # Generate answer
                    messages = create_rag_prompt(context, query)
                    answer = generate_completion(messages)
                    
                    # Display retrieved chunks
                    with st.expander("📚 Retrieved Context"):
                        for i, match in enumerate(results['matches']):
                            st.markdown(f"**Source:** {match['metadata']['source']} (Score: {match['score']:.3f})")
                            st.text(match['metadata']['text'][:300] + "...")
                            st.divider()
                    
                    
                    
                    if answer:
                        st.markdown("### 🎯 Answer:")
                        st.markdown(answer)
                    else:
                        st.error("Failed to generate answer")
                else:
                    st.warning("No relevant documents found")
        else:
            st.warning("Please enter a question")
    
    # Display index stats
    with st.sidebar:
        st.divider()
        stats = get_index_stats(index)
        st.metric("Total Vectors", stats['total_vector_count'])

if __name__ == "__main__":
    main()