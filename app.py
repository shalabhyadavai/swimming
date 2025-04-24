{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;}
{\colortbl;\red255\green255\blue255;\red151\green0\blue126;\red0\green0\blue0;\red13\green100\blue1;
\red181\green0\blue19;\red20\green0\blue196;\red135\green5\blue129;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c66667\c5098\c56863;\csgray\c0;\cssrgb\c0\c45490\c0;
\cssrgb\c76863\c10196\c8627;\cssrgb\c10980\c0\c81176;\cssrgb\c60784\c13725\c57647;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\fs26 \cf2 import\cf3  streamlit \cf2 as\cf3  st\
\cf2 import\cf3  pandas \cf2 as\cf3  pd\
\cf2 import\cf3  numpy \cf2 as\cf3  np\
\cf2 import\cf3  faiss\
\cf2 from\cf3  sentence_transformers \cf2 import\cf3  SentenceTransformer\
\cf2 from\cf3  groq \cf2 import\cf3  Groq\
\cf2 import\cf3  os\
\
\cf4 # App title and layout\cf3 \
st.set_page_config(page_title=\cf5 "SwimCoach AI Assistant"\cf3 , layout=\cf5 "wide"\cf3 )\
st.title(\cf5 "\uc0\u55356 \u57290  SwimCoach AI Assistant"\cf3 )\
st.markdown(\cf5 "Get personalized video recommendations for swimmer training using GenAI."\cf3 )\
\
\cf4 # Sidebar \'96 Athlete profile input\cf3 \
\cf2 with\cf3  st.sidebar:\
    st.header(\cf5 "Athlete Profile"\cf3 )\
    age = st.number_input(\cf5 "Age"\cf3 , min_value=\cf6 8\cf3 , max_value=\cf6 25\cf3 , value=\cf6 14\cf3 )\
    skill = st.selectbox(\cf5 "Skill Level"\cf3 , [\cf5 "Beginner"\cf3 , \cf5 "Intermediate"\cf3 , \cf5 "Advanced"\cf3 ])\
    stroke = st.selectbox(\cf5 "Stroke"\cf3 , [\cf5 "Freestyle"\cf3 , \cf5 "Butterfly"\cf3 , \cf5 "Backstroke"\cf3 , \cf5 "Breaststroke"\cf3 ])\
    goal = st.text_input(\cf5 "Training Goal"\cf3 , \cf5 "Improve breathing rhythm"\cf3 )\
    api_key = st.text_input(\cf5 "Enter your Groq API Key"\cf3 , type=\cf5 "password"\cf3 )\
\
\cf4 # File uploader for custom KB\cf3 \
st.sidebar.markdown(\cf5 "---"\cf3 )\
kb_file = st.sidebar.file_uploader(\cf5 "Upload your KB CSV (Optional)"\cf3 , type=[\cf5 "csv"\cf3 ])\
\
\cf4 # Load KB and model\cf3 \
\cf7 @st.cache_data\cf3 \
\cf2 def\cf3  load_kb(default=\cf2 True\cf3 ):\
    \cf2 if\cf3  default:\
        \cf2 return\cf3  pd.read_csv(\cf5 "swimming_video_kb.csv"\cf3 )\
    \cf2 else\cf3 :\
        \cf2 return\cf3  pd.read_csv(kb_file)\
\
\cf7 @st.cache_resource\cf3 \
\cf2 def\cf3  load_model():\
    \cf2 return\cf3  SentenceTransformer(\cf5 "all-MiniLM-L6-v2"\cf3 )\
\
\cf7 @st.cache_resource\cf3 \
\cf2 def\cf3  build_index(embeddings):\
    dim = embeddings.shape[\cf6 1\cf3 ]\
    index = faiss.IndexFlatL2(dim)\
    index.add(embeddings)\
    \cf2 return\cf3  index\
\
\cf4 # When user clicks the button\cf3 \
\cf2 if\cf3  st.button(\cf5 "Find Recommended Videos"\cf3 ):\
    \cf2 if\cf3  \cf2 not\cf3  api_key:\
        st.warning(\cf5 "Please enter your Groq API key."\cf3 )\
    \cf2 else\cf3 :\
        \cf2 try\cf3 :\
            os.environ[\cf5 "GROQ_API_KEY"\cf3 ] = api_key\
            client = Groq(api_key=api_key)\
\
            \cf4 # Load KB and Model\cf3 \
            df = load_kb(default=(kb_file \cf2 is\cf3  \cf2 None\cf3 ))\
            model = load_model()\
\
            \cf4 # Create embeddings\cf3 \
            df[\cf5 "text"\cf3 ] = df[\cf5 "description"\cf3 ] + \cf5 " "\cf3  + df[\cf5 "transcript"\cf3 ]\
            df[\cf5 "embedding"\cf3 ] = df[\cf5 "text"\cf3 ].apply(\cf2 lambda\cf3  x: model.encode(x).tolist())\
            embeddings = np.array(df[\cf5 "embedding"\cf3 ].tolist()).astype(\cf5 "float32"\cf3 )\
\
            \cf4 # Build index and query\cf3 \
            index = build_index(embeddings)\
            query = \cf5 f"\cf8 \{stroke\}\cf5  \cf8 \{skill\}\cf5  swimmer, age \cf8 \{age\}\cf5 , wants to \cf8 \{goal\}\cf5 "\cf3 \
            query_vector = model.encode([query]).astype(\cf5 "float32"\cf3 )\
            _, I = index.search(query_vector, k=\cf6 3\cf3 )\
            top_entries = df.iloc[I[\cf6 0\cf3 ]]\
            context = \cf5 "\\n\\n"\cf3 .join(top_entries[\cf5 "text"\cf3 ].tolist())\
\
            \cf4 # Prompt for Groq\cf3 \
            prompt = \cf5 f"""\
            You are a professional swimming coach assistant.\
\
            Athlete Profile:\
            - Age: \cf8 \{age\}\cf5 \
            - Skill Level: \cf8 \{skill\}\cf5 \
            - Stroke: \cf8 \{stroke\}\cf5 \
            - Goal: \cf8 \{goal\}\cf5 \
\
            Based on the following training video content, suggest 2\'963 YouTube videos that are likely to help.\
            For each:\
            1. Title (based on content)\
            2. Why it's relevant\
            3. Expected improvements\
\
            Context:\
            \cf8 \{context\}\cf5 \
            """\cf3 \
\
            \cf2 with\cf3  st.spinner(\cf5 "Thinking like a coach..."\cf3 ):\
                response = client.chat.completions.create(\
                    model=\cf5 "llama3-8b-8192"\cf3 ,\
                    messages=[\
                        \{\cf5 "role"\cf3 : \cf5 "system"\cf3 , \cf5 "content"\cf3 : \cf5 "You are a helpful swimming coach assistant."\cf3 \},\
                        \{\cf5 "role"\cf3 : \cf5 "user"\cf3 , \cf5 "content"\cf3 : prompt\}\
                    ]\
                )\
                st.subheader(\cf5 "\uc0\u55356 \u57285  Recommended Videos"\cf3 )\
                st.write(response.choices[\cf6 0\cf3 ].message.content)\
\
        \cf2 except\cf3  Exception \cf2 as\cf3  e:\
            st.error(\cf5 f"\uc0\u10060  Error: \cf8 \{str(e)\}\cf5 "\cf3 )}