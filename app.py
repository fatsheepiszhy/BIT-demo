import streamlit as st
import os
import shutil
import uuid  # <--- 新增：用于生成随机文件夹名
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ================= 配置区域 =================
# 1. SiliconFlow Key
SILICON_FLOW_API_KEY = "sk-ddnnkphyxoiwaszzcomttqjfspitnoqevwbwaxhwwcyxyctj"

# 2. DeepSeek Key
DEEPSEEK_API_KEY = "sk-f3a1469735ec4c81b4bc63a1618501d0"

# 3. 数据文件夹
DATA_FOLDER = "Data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ================= 页面基础设置 =================
st.set_page_config(page_title="RAG 智能问答系统", page_icon="🤖", layout="wide")
st.title("RAG 智能问答系统")
st.markdown("### 基于 DeepSeek + ChromaDB 的垂直领域知识问答")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 【核心修改点 1】初始化数据库路径
# 我们把数据库路径存在 session_state 里，而不是写死
if "current_db_path" not in st.session_state:
    st.session_state.current_db_path = "./chroma_db_init" # 初始默认路径

# ================= 侧边栏：知识库管理 =================
with st.sidebar:
    st.header("📚 知识库管理")
    
    # 1. 文件上传
    uploaded_file = st.file_uploader("上传 PDF 文档 (如果不传则使用默认)", type=["pdf"])
    
    if uploaded_file:
        current_pdf_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(current_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已加载上传文件: {uploaded_file.name}")
    else:
        current_pdf_path = os.path.join(DATA_FOLDER, "test.pdf")
        if os.path.exists(current_pdf_path):
            st.info(f"使用默认文件: test.pdf(是关于英语口语模板的)")
        else:
            st.warning("⚠️ 默认文件不存在，请上传 PDF！")

    # 2. 构建数据库按钮
    if st.button("🔄 初始化/更新 知识库"):
        if not os.path.exists(current_pdf_path):
            st.error("找不到文件，无法构建！")
        else:
            with st.status("正在构建知识库...", expanded=True) as status:
                try:
                    # --- Step 1: 读取与切分 ---
                    st.write(f"1. 正在读取文件: {os.path.basename(current_pdf_path)}...")
                    loader = PyMuPDFLoader(current_pdf_path)
                    docs = loader.load()
                    
                    st.write("2. 正在切分文本 (Chunking)...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", " ", "。", "，"]
                    )
                    chunks = text_splitter.split_documents(docs)

                    # --- Step 2: 向量化与存储 ---
                    st.write("3. 正在初始化 Embedding 模型...")
                    embeddings = OpenAIEmbeddings(
                        model="BAAI/bge-m3",
                        api_key=SILICON_FLOW_API_KEY,
                        base_url="https://api.siliconflow.cn/v1",
                        chunk_size=50 # 限制批次大小，防止报错
                    )

                    st.write("4. 正在构建 ChromaDB...")
                    new_db_path = f"./chroma_db_{uuid.uuid4().hex[:8]}"
                    
                    # 尝试清理旧
                    if os.path.exists(st.session_state.current_db_path):
                        try:
                            shutil.rmtree(st.session_state.current_db_path)
                        except:
                            pass # 忽略删除错误
                    
                    # 存入新路径
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=new_db_path
                    )
                    
                    # 更新 session_state 里的路径
                    st.session_state.current_db_path = new_db_path
                    
                    status.update(label="✅ 知识库构建成功！", state="complete", expanded=False)
                    st.success(f"知识库已更新，请开始提问！")
                    
                except Exception as e:
                    st.error(f"构建失败: {e}")

# ================= 主界面：聊天逻辑 =================

# 1. 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 接收用户输入
user_input = st.chat_input("请输入关于文档的问题...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # 【核心修改点 3】检查当前路径是否存在
        if not os.path.exists(st.session_state.current_db_path):
            st.error("❌ 数据库未找到！请先点击左侧侧边栏的【初始化/更新 知识库】按钮。")
            st.stop()
            
        status_box = st.empty()
        status_box.markdown("🤔 正在检索并思考...")

        try:
            # 1. 准备 Embedding
            embedding_model = OpenAIEmbeddings(
                model="BAAI/bge-m3",
                api_key=SILICON_FLOW_API_KEY,
                base_url="https://api.siliconflow.cn/v1",
                chunk_size=50
            )
            
            # 2. 加载数据库 (使用 session_state 里的动态路径)
            vectordb = Chroma(
                persist_directory=st.session_state.current_db_path, 
                embedding_function=embedding_model
            )
            
            # 3. 检索
            results = vectordb.similarity_search(user_input, k=5)
            
            if not results:
                answer = "❌ 未在文档中找到相关信息。"
                references = []
            else:
                context_parts = []
                references = []
                for i, res in enumerate(results):
                    page = res.metadata.get("page", 0) + 1
                    context_parts.append(f"[证据{i+1} (第{page}页)]: {res.page_content}")
                    references.append(f"**[证据 {i+1}] 第 {page} 页**: {res.page_content}")
                
                context_str = "\n\n".join(context_parts)
                
                prompt = f"""
                你是一个严谨的面试助手。请根据下面的【背景信息】回答用户的【问题】。
                
                要求：
                1. 回答必须完全基于背景信息。
                2. 在回答的关键句后面，请注明参考了哪个证据（例如：[证据1]）。
                3. 如果背景信息不够回答问题，请直接说“资料不足”。

                【背景信息】：
                {context_str}

                【问题】：
                {user_input}
                """
                
                llm = ChatOpenAI(
                    model="deepseek-chat", 
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com",
                    temperature=0.3
                )
                
                response = llm.invoke(prompt)
                answer = response.content

            status_box.markdown(answer)
            
            if references:
                with st.expander("📖 查看检索到的原文证据"):
                    for ref in references:
                        st.markdown(ref)
                        st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"发生错误: {e}")

