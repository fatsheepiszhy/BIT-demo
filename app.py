import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# 尝试从 Streamlit Secrets 读取，如果本地运行没有配置 secrets，则使用空字符串或环境变量
if "SILICON_KEY" in st.secrets:
    SILICON_FLOW_API_KEY = st.secrets["SILICON_KEY"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_KEY"]
# 3. 数据库路径
DB_PATH = "./chroma_db_data"

# 4. 默认数据文件夹
DATA_FOLDER = "Data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ================= 页面基础设置 =================
st.set_page_config(page_title="RAG 智能面试助手", page_icon="🤖", layout="wide")
st.title("🤖 北理工复试助手 (RAG System)")
st.markdown("### 基于 DeepSeek + ChromaDB 的垂直领域知识问答")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 侧边栏：知识库管理 =================
with st.sidebar:
    st.header("📚 知识库管理")
    
    # 1. 文件上传
    uploaded_file = st.file_uploader("上传 PDF 文档 (如果不传则使用默认)", type=["pdf"])
    
    # 确定当前使用的 PDF 路径
    if uploaded_file:
        # 如果用户上传了文件，保存到本地
        current_pdf_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(current_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已加载上传文件: {uploaded_file.name}")
    else:
        # 否则使用默认文件
        current_pdf_path = os.path.join(DATA_FOLDER, "test.pdf")
        if os.path.exists(current_pdf_path):
            st.info(f"使用默认文件: test.pdf")
        else:
            st.warning("⚠️ 默认文件不存在，请上传 PDF！")

    # 2. 构建数据库按钮 (对应 Step 1 & Step 2)
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
                    st.write(f"   - 文档加载完成，共 {len(docs)} 页")
                    
                    st.write("2. 正在切分文本 (Chunking)...")
                    # 复用你的参数：chunk_size=500, chunk_overlap=200
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", " ", "。", "，"]
                    )
                    chunks = text_splitter.split_documents(docs)
                    st.write(f"   - 切分完成，共 {len(chunks)} 个碎片")

                    # --- Step 2: 向量化与存储 ---
                    st.write("3. 正在初始化 Embedding 模型 (BAAI/bge-m3)...")
                    embeddings = OpenAIEmbeddings(
                        model="BAAI/bge-m3",
                        api_key=SILICON_FLOW_API_KEY,
                        base_url="https://api.siliconflow.cn/v1"
                    )

                    st.write("4. 正在重建 ChromaDB 数据库...")
                    # 为了保证数据最新，先清理旧库
                    if os.path.exists(DB_PATH):
                        shutil.rmtree(DB_PATH)
                    
                    # 存入
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=DB_PATH
                    )
                    
                    status.update(label="✅ 知识库构建成功！", state="complete", expanded=False)
                    st.success("知识库已更新，请开始提问！")
                    
                except Exception as e:
                    st.error(f"构建失败: {e}")

# ================= 主界面：聊天逻辑 (对应 Step 3) =================

# 1. 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 接收用户输入
user_input = st.chat_input("请输入关于文档的问题（例如：关于兴趣爱好的模板是什么？）")

if user_input:
    # A. 存入并展示用户问题
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # B. AI 回答
    with st.chat_message("assistant"):
        # 检查数据库是否存在
        if not os.path.exists(DB_PATH):
            st.error("❌ 数据库未找到！请先点击左侧侧边栏的【初始化/更新 知识库】按钮。")
            st.stop()
            
        status_box = st.empty()
        status_box.markdown("🤔 正在检索并思考...")

        try:
            # --- 1. 准备 Embedding ---
            embedding_model = OpenAIEmbeddings(
                model="BAAI/bge-m3",
                api_key=SILICON_FLOW_API_KEY,
                base_url="https://api.siliconflow.cn/v1"
            )
            
            # --- 2. 加载数据库 ---
            vectordb = Chroma(
                persist_directory=DB_PATH, 
                embedding_function=embedding_model
            )
            
            # --- 3. 检索 (复用你的 k=10) ---
            results = vectordb.similarity_search(user_input, k=10)
            
            if not results:
                answer = "❌ 未在文档中找到相关信息。"
                references = []
            else:
                # --- 4. 拼接上下文与证据 ---
                context_parts = []
                references = []
                
                for i, res in enumerate(results):
                    page = res.metadata.get("page", 0) + 1
                    # 准备给大模型看的内容
                    context_parts.append(f"[证据{i+1} (第{page}页)]: {res.page_content}")
                    # 准备展示给用户看的引用信息
                    references.append(f"**[证据 {i+1}] 第 {page} 页**: {res.page_content}")
                
                context_str = "\n\n".join(context_parts)
                
                # --- 5. 生成回答 (复用你的 Prompt) ---
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

            # C. 展示结果
            status_box.markdown(answer)
            
            # D. 展示引用来源 (折叠效果，看起来很高级)
            if references:
                with st.expander("📖 查看检索到的原文证据 (Source Context)"):
                    for ref in references:
                        st.markdown(ref)
                        st.divider()

            # E. 保存 AI 回答到历史记录
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:

            st.error(f"发生错误: {e}")
