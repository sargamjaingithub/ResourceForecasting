import streamlit as st

st.title("Test App")
st.write("If you see this, basic Streamlit works!")

try:
    import pandas as pd
    st.success("✅ Pandas imported")
except Exception as e:
    st.error(f"❌ Pandas error: {e}")

try:
    import numpy as np
    st.success("✅ Numpy imported")
except Exception as e:
    st.error(f"❌ Numpy error: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    st.success("✅ Matplotlib imported")
except Exception as e:
    st.error(f"❌ Matplotlib error: {e}")

st.write("All imports successful!")
