import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import gc  
import re  
from tqdm import tqdm 
import numpy as np

# 1. 页面基本配置 (必须放在脚本最开头)
st.set_page_config(
    page_title="Wavelength Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 优化后的 CSS 样式 (去掉了破坏响应式布局的全局设定，保留卡片美化)
st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
        background-color: #f0f4f8;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .header-container h1 {
        color: #1E3A8A;
        font-size: 2.2rem;
        margin-bottom: 15px;
    }
    .header-container p {
        color: #475569;
        font-size: 1.1rem;
        margin: 0;
    }
    .molecule-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 15px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. 页面标题和简介 (使用居中卡片样式)
st.markdown(
    """
    <div class='header-container'>
        <h1>🌟 Organic UV-Vis & Fluorescence Wavelength Predictor</h1>
        <p>Predict both absorption and emission wavelengths of organic molecules based on their structure (SMILES) and solvent.</p>
        <p style="font-size: 0.9rem; margin-top: 10px;">
            Code and data: <a href='https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction' target='_blank'>GitHub Repository</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 字典与函数定义保持不变 ---
solvent_data = {
    "H2O": {"Et30": 63.1, "SP": 0.681, "SdP": 0.997, "SA": 1.062, "SB": 0.025},
    "MeOH": {"Et30": 55.4, "SP": 0.608, "SdP": 0.904, "SA": 0.605, "SB": 0.545},
    "EtOH": {"Et30": 51.9, "SP": 0.633, "SdP": 0.783, "SA": 0.4, "SB": 0.658},
    "MeCN": {"Et30": 45.6, "SP": 0.645, "SdP": 0.974, "SA": 0.044, "SB": 0.286},
    "DMSO": {"Et30": 45.1, "SP": 0.83, "SdP": 1.0, "SA": 0.072, "SB": 0.647},
    "THF": {"Et30": 37.4, "SP": 0.714, "SdP": 0.634, "SA": 0.0, "SB": 0.591},
    "CH2Cl2": {"Et30": 40.7, "SP": 0.761, "SdP": 0.769, "SA": 0.04, "SB": 0.178},
    "CHCl3": {"Et30": 39.1, "SP": 0.783, "SdP": 0.614, "SA": 0.047, "SB": 0.071},
    "CCl4": {"Et30": 32.4, "SP": 0.768, "SdP": 0.0, "SA": 0.0, "SB": 0.044},
    "DMF": {"Et30": 43.2, "SP": 0.759, "SdP": 0.977, "SA": 0.031, "SB": 0.613},
    "DCE": {"Et30": 41.3, "SP": 0.771, "SdP": 0.742, "SA": 0.03, "SB": 0.126},
    "CS2": {"Et30": 32.8, "SP": 1.0, "SdP": 0.0, "SA": 0.0, "SB": 0.104},
    "1,4-dioxane": {"Et30": 36.0, "SP": 0.737, "SdP": 0.312, "SA": 0.0, "SB": 0.444},
    "Toluene": {"Et30": 33.9, "SP": 0.782, "SdP": 0.284, "SA": 0.0, "SB": 0.128},
    "Glycerol": {"Et30": 57.0, "SP": 0.828, "SdP": 0.921, "SA": 0.653, "SB": 0.309},
    "Hexane": {"Et30": 31.0, "SP": 0.616, "SdP": 0.0, "SA": 0.0, "SB": 0.056},
    "Acetic acid": {"Et30": 51.7, "SP": 0.651, "SdP": 0.676, "SA": 0.689, "SB": 0.39},
    "Acetone": {"Et30": 42.2, "SP": 0.651, "SdP": 0.907, "SA": 0.0, "SB": 0.475},
    "Ethyl acetate": {"Et30": 38.1, "SP": 0.656, "SdP": 0.603, "SA": 0.0, "SB": 0.542},
    "Benzene": {"Et30": 34.3, "SP": 0.793, "SdP": 0.27, "SA": 0.0, "SB": 0.124},
    "Cyclohexane": {"Et30": 30.9, "SP": 0.683, "SdP": 0.0, "SA": 0.0, "SB": 0.073},
    "Nitroethane": {"Et30": 43.6, "SP": 0.706, "SdP": 0.902, "SA": 0.0, "SB": 0.234},
    "Chlorobenzene": {"Et30": 36.8, "SP": 0.833, "SdP": 0.537, "SA": 0.0, "SB": 0.182},
    "Fluorobenzene": {"Et30": 37.0, "SP": 0.761, "SdP": 0.511, "SA": 0.0, "SB": 0.113},
    "Nitrobenzene": {"Et30": 41.2, "SP": 0.891, "SdP": 0.873, "SA": 0.056, "SB": 0.24},
    "Cyclohexanone": {"Et30": 39.8, "SP": 0.766, "SdP": 0.745, "SA": 0.0, "SB": 0.482},
    "Ethylene glycol": {"Et30": 56.3, "SP": 0.777, "SdP": 0.91, "SA": 0.717, "SB": 0.534},
    "Propionitrile": {"Et30": 43.6, "SP": 0.668, "SdP": 0.888, "SA": 0.03, "SB": 0.365},
    "Ethyl formate": {"Et30": 40.9, "SP": 0.648, "SdP": 0.707, "SA": 0.0, "SB": 0.477},
    "Methyl acetate": {"Et30": 38.9, "SP": 0.645, "SdP": 0.637, "SA": 0.0, "SB": 0.527},
    "Dimethyl carbonate": {"Et30": 38.2, "SP": 0.653, "SdP": 0.531, "SA": 0.064, "SB": 0.433},
    "1-propanol": {"Et30": 50.7, "SP": 0.658, "SdP": 0.748, "SA": 0.367, "SB": 0.782},
    "2-propanol": {"Et30": 48.4, "SP": 0.633, "SdP": 0.808, "SA": 0.283, "SB": 0.83},
    "1,2-propanediol": {"Et30": 54.1, "SP": 0.731, "SdP": 0.888, "SA": 0.475, "SB": 0.598},
    "Benzonitrile": {"Et30": 41.5, "SP": 0.851, "SdP": 0.852, "SA": 0.047, "SB": 0.281},
    "Heptane": {"Et30": 31.1, "SP": 0.635, "SdP": 0.0, "SA": 0.0, "SB": 0.083},
    "1-heptanol": {"Et30": 48.5, "SP": 0.706, "SdP": 0.499, "SA": 0.302, "SB": 0.912},
    "p-xylene": {"Et30": 33.1, "SP": 0.778, "SdP": 0.175, "SA": 0.0, "SB": 0.16},
    "Ethoxybenzene": {"Et30": 36.6, "SP": 0.81, "SdP": 0.669, "SA": 0.0, "SB": 0.295},
    "Octane": {"Et30": 31.1, "SP": 0.65, "SdP": 0.0, "SA": 0.0, "SB": 0.079},
    "Dibutyl ether": {"Et30": 33.0, "SP": 0.672, "SdP": 0.175, "SA": 0.0, "SB": 0.637},
    "Mesitylene": {"Et30": 32.9, "SP": 0.775, "SdP": 0.155, "SA": 0.0, "SB": 0.19},
    "Nonane": {"Et30": 31.0, "SP": 0.66, "SdP": 0.0, "SA": 0.0, "SB": 0.053},
    "1,2,3,4-tetrahydronaphthalene": {"Et30": 33.5, "SP": 0.838, "SdP": 0.182, "SA": 0.0, "SB": 0.18},
    "Decane": {"Et30": 31.0, "SP": 0.669, "SdP": 0.0, "SA": 0.0, "SB": 0.066},
    "1-decanol": {"Et30": 47.7, "SP": 0.722, "SdP": 0.383, "SA": 0.259, "SB": 0.912},
    "1-methylnaphthalene": {"Et30": 35.3, "SP": 0.908, "SdP": 0.51, "SA": 0.0, "SB": 0.156},
    "Dodecane": {"Et30": 31.1, "SP": 0.683, "SdP": 0.0, "SA": 0.0, "SB": 0.086},
    "Tributylamine": {"Et30": 32.1, "SP": 0.689, "SdP": 0.06, "SA": 0.0, "SB": 0.854},
    "Dibenzyl ether": {"Et30": 36.3, "SP": 0.877, "SdP": 0.509, "SA": 0.0, "SB": 0.33},
    "Trimethyl phosphate": {"Et30": 43.6, "SP": 0.707, "SdP": 0.909, "SA": 0.0, "SB": 0.522},
    "Butanenitrile": {"Et30": 42.5, "SP": 0.689, "SdP": 0.864, "SA": 0.0, "SB": 0.384},
    "2-butanone": {"Et30": 41.3, "SP": 0.669, "SdP": 0.872, "SA": 0.0, "SB": 0.52},
    "Sulfolane": {"Et30": 44.0, "SP": 0.83, "SdP": 0.896, "SA": 0.052, "SB": 0.365},
    "1-butanol": {"Et30": 49.7, "SP": 0.674, "SdP": 0.655, "SA": 0.341, "SB": 0.809},
    "Ethyl ether": {"Et30": 34.5, "SP": 0.617, "SdP": 0.385, "SA": 0.0, "SB": 0.562},
    "2-methyl-2-propanol": {"Et30": 43.3, "SP": 0.632, "SdP": 0.732, "SA": 0.145, "SB": 0.928},
    "1,2-dimethoxyethane": {"Et30": 38.2, "SP": 0.68, "SdP": 0.625, "SA": 0.0, "SB": 0.636},
    "Pyridine": {"Et30": 40.5, "SP": 0.842, "SdP": 0.761, "SA": 0.033, "SB": 0.581},
    "1-methyl-2-pyrrolidinone": {"Et30": 48.0, "SP": 0.812, "SdP": 0.959, "SA": 0.024, "SB": 0.613},
    "2-methyltetrahydrofuran": {"Et30": 36.5, "SP": 0.7, "SdP": 0.768, "SA": 0.0, "SB": 0.584},
    "2-pentanone": {"Et30": 41.1, "SP": 0.689, "SdP": 0.783, "SA": 0.01, "SB": 0.537},
    "3-pentanone": {"Et30": 39.3, "SP": 0.692, "SdP": 0.785, "SA": 0.0, "SB": 0.557},
    "Propyl acetate": {"Et30": 37.5, "SP": 0.67, "SdP": 0.559, "SA": 0.0, "SB": 0.548},
    "Piperidine": {"Et30": 35.5, "SP": 0.754, "SdP": 0.365, "SA": 0.0, "SB": 0.933},
    "2-methylbutane": {"Et30": 30.9, "SP": 0.581, "SdP": 0.0, "SA": 0.0, "SB": 0.053},
    "Pentane": {"Et30": 31.0, "SP": 0.593, "SdP": 0.0, "SA": 0.0, "SB": 0.073},
    "Tert-butyl methyl ether": {"Et30": 34.7, "SP": 0.622, "SdP": 0.422, "SA": 0.0, "SB": 0.567},
    "Hexafluorobenzene": {"Et30": 34.2, "SP": 0.623, "SdP": 0.252, "SA": 0.0, "SB": 0.119},
    "1,2-dichlorobenzene": {"Et30": 38.0, "SP": 0.869, "SdP": 0.676, "SA": 0.033, "SB": 0.144},
    "Bromobenzene": {"Et30": 36.6, "SP": 0.875, "SdP": 0.497, "SA": 0.0, "SB": 0.192},
    "N,N-dimethylacetamide": {"Et30": 42.9, "SP": 0.763, "SdP": 0.987, "SA": 0.028, "SB": 0.65},
    "Butyl acetate": {"Et30": 38.5, "SP": 0.674, "SdP": 0.535, "SA": 0.0, "SB": 0.525},
    "N,N-diethylacetamide": {"Et30": 41.4, "SP": 0.748, "SdP": 0.918, "SA": 0.0, "SB": 0.66},
    "Diisopropyl ether": {"Et30": 34.1, "SP": 0.625, "SdP": 0.324, "SA": 0.0, "SB": 0.657},
    "Dipropyl ether": {"Et30": 34.0, "SP": 0.645, "SdP": 0.286, "SA": 0.0, "SB": 0.666},
    "1-hexanol": {"Et30": 48.8, "SP": 0.698, "SdP": 0.552, "SA": 0.315, "SB": 0.879},
    "Hexamethylphosphoramine": {"Et30": 40.9, "SP": 0.744, "SdP": 1.1, "SA": 0.0, "SB": 0.813},
    "Triethylamine": {"Et30": 32.1, "SP": 0.66, "SdP": 0.108, "SA": 0.0, "SB": 0.885},
    "(Trifluoromethyl)benzene": {"Et30": 38.5, "SP": 0.694, "SdP": 0.663, "SA": 0.014, "SB": 0.073},
    "1,1,2-trichlorotrifluoroethane": {"Et30": 33.2, "SP": 0.596, "SdP": 0.152, "SA": 0.0, "SB": 0.038},
    "1,1,2,2-tetrachloroethane": {"Et30": 39.4, "SP": 0.845, "SdP": 0.792, "SA": 0.0, "SB": 0.017},
    "1,1,1-trichloroethane": {"Et30": 36.2, "SP": 0.737, "SdP": 0.5, "SA": 0.0, "SB": 0.085},
    "2,2,2-trifluoroethanol": {"Et30": 59.8, "SP": 0.543, "SdP": 0.922, "SA": 0.893, "SB": 0.107},
    "dioxane": {"Et30": 36.0, "SP": 0.737, "SdP": 0.312, "SA": 0.0, "SB": 0.444},
    "Me-THF": {"Et30": 36.5, "SP": 0.7, "SdP": 0.768, "SA": 0.0, "SB": 0.584},
    "DCB": {"Et30": 38.0, "SP": 0.869, "SdP": 0.676, "SA": 0.033, "SB": 0.144},
    "DMA": {"Et30": 42.9, "SP": 0.763, "SdP": 0.987, "SA": 0.028, "SB": 0.65},
    "N-Methylformamide": {"Et30": 54.1, "SP": 0.759, "SdP": 0.977, "SA": 0.031, "SB": 0.613}
}

features_abs = ['nBondsD', 'NumAliphaticHeterocycles', 'PEOE_VSA8', 'SdssC', "VSA_EState2", "SlogP_VSA10", "SMR_VSA3","SMR_VSA10"]
features_em = ["nBondsD", "SdssC", "PEOE_VSA8", "SMR_VSA3", "n6HRing", "SMR_VSA10"]
all_required_features = list(set(features_abs + features_em))

@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor_abs():
    return TabularPredictor.load("./ag-20250620_005406")

@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor_em():
    return TabularPredictor.load("./ag-20250609_005753")

def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    draw_options = d2d.drawOptions()
    draw_options.background = '#ffffff' # 改为纯白背景更好看
    draw_options.padding = 0.0
    draw_options.additionalBondPadding = 0.0
    draw_options.annotationFontScale = 1.0
    draw_options.addAtomIndices = False
    draw_options.addStereoAnnotation = False
    draw_options.bondLineWidth = 1.5
    draw_options.includeMetadata = False
    
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    
    svg = re.sub(r'<rect [^>]*stroke:black[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect [^>]*stroke:#000000[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect[^>]*/>', '', svg, flags=re.DOTALL)
    if 'viewBox' in svg:
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {size[0]} {size[1]}"', svg)
    
    return svg

def calc_rdkit_descriptors(smiles_list):
    desc_names = [desc_name for desc_name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    results = []
    valid_indices = []
    for idx, smi in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Calculating RDKit descriptors"):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            descriptors = calculator.CalcDescriptors(mol)
            processed_descriptors = []
            for val in descriptors:
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    processed_descriptors.append(np.nan)
                elif val is None:  
                    processed_descriptors.append(np.nan)
                else:
                    processed_descriptors.append(val)
            results.append(processed_descriptors)
            valid_indices.append(idx)
        except Exception:
            continue
    return pd.DataFrame(results, columns=desc_names, index=valid_indices)

def calc_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    valid_smiles = []
    for smi in tqdm(smiles_list, desc="Calculating Mordred descriptors"):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            res = calc(mol)
            descriptor_dict = {}
            for key, val in res.asdict().items():
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    descriptor_dict[key] = np.nan
                elif val is None:
                    descriptor_dict[key] = np.nan
                elif hasattr(val, '__class__') and val.__class__.__name__ == 'Missing': 
                    descriptor_dict[key] = np.nan
                else:
                    descriptor_dict[key] = val
            results.append(descriptor_dict)
            valid_smiles.append(smi)
        except Exception:
            continue
    df_mordred = pd.DataFrame(results)
    df_mordred['SMILES'] = valid_smiles
    return df_mordred

def merge_features_without_duplicates(original_df, *feature_dfs):
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

# ==========================================
# 4. 界面布局：侧边栏 (Sidebar) 作为输入区
# ==========================================
with st.sidebar:
    st.header("⚙️ Input Parameters")
    st.markdown("---")
    
    smiles = st.text_input(
        "1. Enter SMILES:", 
        placeholder="e.g., [BH3-][P+]1(c2ccccc2)c2ccccc2-c2sc3ccccc3c21"
    )
    
    solvent = st.selectbox(
        "2. Select Solvent:", 
        list(solvent_data.keys())
    )
    
    st.markdown("---")
    submit_button = st.button("🚀 Predict Now", type="primary", use_container_width=True)


# ==========================================
# 5. 界面布局：主干区域展示结果
# ==========================================
if submit_button:
    if not smiles:
        st.warning("⚠️ Please enter a valid SMILES string in the sidebar.")
    elif not solvent:
        st.warning("⚠️ Please select a solvent in the sidebar.")
    else:
        with st.spinner("Processing molecule and making predictions..."):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    st.error("❌ Invalid SMILES input. Please check the format.")
                    st.stop()
                
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)

                # 计算特征
                solvent_params = solvent_data[solvent]
                smiles_list = [smiles] 
                rdkit_features = calc_rdkit_descriptors(smiles_list)
                mordred_features = calc_mordred_descriptors(smiles_list)
                merged_features = merge_features_without_duplicates(rdkit_features, mordred_features)
                data = merged_features.loc[:, all_required_features]

                base_solvent_dict = {
                    "Et30": [solvent_params["Et30"]],
                    "SP": [solvent_params["SP"]],
                    "SdP": [solvent_params["SdP"]],
                    "SA": [solvent_params["SA"]],
                    "SB": [solvent_params["SB"]]
                }
                
                predict_dict_abs = base_solvent_dict.copy()
                for feat in features_abs:
                    predict_dict_abs[feat] = [data.iloc[0][feat]]
                predict_df_abs = pd.DataFrame(predict_dict_abs)
                
                predict_dict_em = base_solvent_dict.copy()
                for feat in features_em:
                    predict_dict_em[feat] = [data.iloc[0][feat]]
                predict_df_em = pd.DataFrame(predict_dict_em)

                try:
                    # 加载模型并预测
                    predictor_abs = load_predictor_abs()
                    predictor_em = load_predictor_em()
                    
                    predict_df_abs_1 = pd.concat([predict_df_abs, predict_df_abs], axis=0)
                    predict_df_em_1 = pd.concat([predict_df_em, predict_df_em], axis=0)
                    
                    try:
                        pred_abs = predictor_abs.predict(predict_df_abs_1)
                        res_abs = f"{int(pred_abs.iloc[0])} nm"
                    except Exception as e:
                        res_abs = "Error"
                        st.warning(f"Absorption model prediction failed: {str(e)}")

                    try:
                        pred_em = predictor_em.predict(predict_df_em_1)
                        res_em = f"{int(pred_em.iloc[0])} nm"
                    except Exception as e:
                        res_em = "Error"
                        st.warning(f"Emission model prediction failed: {str(e)}")

                    st.success("✅ Prediction completed successfully!")

                    # --- 核心视觉展示区：左右分栏 ---
                    col1, col2 = st.columns([1, 1.5], gap="large")
                    
                    with col1:
                        st.subheader("🧬 Molecule Structure")
                        svg = mol_to_image(mol)
                        st.markdown(
                            f'<div class="molecule-container">{svg}</div>', 
                            unsafe_allow_html=True
                        )
                        mol_weight = Descriptors.MolWt(mol)
                        st.caption(f"**Molecular Weight:** {mol_weight:.2f} g/mol")

                    with col2:
                        st.subheader("📊 Prediction Results")
                        st.markdown(f"**Solvent:** `{solvent}`")
                        
                        # 使用强大的 st.metric 组件展示核心数据
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric(label="🔵 Absorption Wavelength", value=res_abs)
                        with m_col2:
                            st.metric(label="🟢 Emission Wavelength", value=res_em)
                            
                    st.markdown("---")
                    
                    # 将枯燥的数据表格收纳进 expander 中
                    with st.expander("🔍 View Technical Details & Extracted Features"):
                        st.write("Base Solvent Features:")
                        st.dataframe(pd.DataFrame(base_solvent_dict), use_container_width=True)
                        st.write("Full Extracted Features for Absorption:")
                        st.dataframe(predict_df_abs, use_container_width=True)
                        st.write("Full Extracted Features for Emission:")
                        st.dataframe(predict_df_em, use_container_width=True)

                    gc.collect()

                except Exception as e:
                    st.error(f"Model loading or execution failed: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    # 刚进入页面时显示的占位提示
    st.info("👈 Please enter the SMILES string and select a solvent in the sidebar to start prediction.")
