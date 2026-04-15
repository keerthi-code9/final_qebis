import streamlit as st
import streamlit.components.v1 as components

with open("qebis_dashboard.html", "r", encoding="utf-8") as f:
    html_data = f.read()

components.html(html_data, height=900, scrolling=True)
