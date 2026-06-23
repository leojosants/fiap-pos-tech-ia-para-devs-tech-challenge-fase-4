"""
debug_secrets.py
=================
Página de diagnóstico temporária para investigar por que GROQ_API_KEY
não está sendo encontrada no deploy do Streamlit Community Cloud.

USO TEMPORÁRIO: rodar isso como app principal (ou importar e chamar a
função no main.py) para inspecionar o estado real de st.secrets e
os.environ no servidor, sem expor a chave real (mostra só o tamanho e
os primeiros caracteres).

REMOVER após o diagnóstico — não deixar em produção.
"""

import os
import streamlit as st

st.title("🔍 Diagnóstico de Secrets")

st.subheader("1. st.secrets — chaves disponíveis")
try:
    keys = list(st.secrets.keys())
    st.write(f"Chaves encontradas em st.secrets: {keys}")
    if "GROQ_API_KEY" in st.secrets:
        val = st.secrets["GROQ_API_KEY"]
        st.success(f"GROQ_API_KEY encontrada em st.secrets! Tamanho: {len(val)}, início: {val[:6]}...")
    else:
        st.error("GROQ_API_KEY NÃO está em st.secrets.")
except Exception as e:
    st.error(f"Erro ao acessar st.secrets: {type(e).__name__}: {e}")

st.subheader("2. os.environ — variável GROQ_API_KEY")
env_val = os.environ.get("GROQ_API_KEY")
if env_val:
    st.success(f"GROQ_API_KEY encontrada em os.environ! Tamanho: {len(env_val)}, início: {env_val[:6]}...")
else:
    st.error("GROQ_API_KEY NÃO está em os.environ.")

st.subheader("3. Todas as variáveis de ambiente que contêm 'GROQ'")
groq_vars = {k: v for k, v in os.environ.items() if "GROQ" in k.upper()}
if groq_vars:
    for k, v in groq_vars.items():
        st.write(f"- {k}: tamanho={len(v)}, início={v[:6]}...")
else:
    st.warning("Nenhuma variável de ambiente contendo 'GROQ' foi encontrada.")

st.subheader("4. Existe arquivo .streamlit/secrets.toml?")
secrets_path = ".streamlit/secrets.toml"
st.write(f"Caminho verificado: {os.path.abspath(secrets_path)}")
st.write(f"Existe: {os.path.exists(secrets_path)}")