import streamlit as st
import streamlit.components.v1 as components
import json

with open("qebis_data.json", "r") as f:
    qebis_data = json.load(f)

with open("qebis_dashboard.html", "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("__QEBIS_DATA__", json.dumps(qebis_data))

components.html(html, height=3000, scrolling=True)
