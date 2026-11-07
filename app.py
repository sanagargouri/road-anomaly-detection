"""
Interface de démonstration – Détection d'obstacles inattendus sur la route (RoadAnomaly)
Lancement :  streamlit run app.py
"""
import os
import json
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib

# ---------------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Détecteur d'obstacles routiers", page_icon="🚧", layout="wide")

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DOSSIER_MODELES = os.environ.get("RA_DOSSIER_MODELES", os.path.join(DOSSIER, "RoadAnomaly_modeles"))
FRAMES_DIR = os.environ.get("RA_FRAMES_DIR", os.path.join(DOSSIER, "data", "RoadAnomaly", "frames"))
POIDS_RESNET = os.environ.get("RA_POIDS_RESNET", "imagenet")
POIDS_RESNET = None if POIDS_RESNET == "none" else POIDS_RESNET
H, W = 720, 1280

# Statistiques de standardisation (étape 17b) : relues depuis config_interface.json si présent
CONFIG_PATH = os.path.join(DOSSIER_MODELES, "config_interface.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"moyenne": [0.455, 0.455, 0.455], "ecart_type": [0.224, 0.22, 0.239], "images_test": []}
MOYENNE = np.array(CONFIG["moyenne"], dtype=np.float32)
ECART_TYPE = np.array(CONFIG["ecart_type"], dtype=np.float32)

# Les 5 modèles du projet : type d'architecture, fichier de poids, scores sur le test (étapes 28b et 28c)
MODELES = {
    "DeepLabV3+ avec ResNet50 (modèle retenu)": ("deeplab", "modele_5_deeplabv3plus",
                                                  {"F1": 0.7954, "Recall": 0.7625, "Precision": 0.8314, "IoU": 0.6604, "FPR": 1.57}),
    "U-Net avec ResNet50": ("unet_resnet", "modele_4_unet_resnet50_gele",
                            {"F1": 0.7220, "Recall": 0.6155, "Precision": 0.8731, "IoU": 0.5649, "FPR": 0.91}),
    "U-Net construit de zéro": ("unet", "grid_lr0.0001_bs2",
                                {"F1": 0.4523, "Recall": 0.5262, "Precision": 0.3966, "IoU": 0.2922, "FPR": 8.11}),
    "U-Net sans augmentation": ("unet", "modele_2_unet_sans_aug",
                                {"F1": 0.3419, "Recall": 0.3033, "Precision": 0.3917, "IoU": 0.2062, "FPR": 4.77}),
    "Encodeur-décodeur type FCN": ("fcn", "modele_3_fcn",
                                   {"F1": 0.3071, "Recall": 0.2719, "Precision": 0.3529, "IoU": 0.1814, "FPR": 5.05}),
}

# ---------------------------------------------------------------------------
# Style : une bande « asphalte » avec un marquage au sol, le reste sobre
# ---------------------------------------------------------------------------
ASPHALTE, BETON, TEXTE_DOUX, MARQUAGE = "#23292E", "#EEF0F1", "#5B6670", "#F2B705"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, button, input, select {{
    font-family: 'Barlow', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}}
.stApp {{ background: {BETON}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
.bandeau {{
    background: {ASPHALTE}; color: #FFFFFF; border-radius: 6px;
    padding: 1.6rem 2rem 0 2rem; margin-bottom: 1.4rem; overflow: hidden;
}}
.bandeau h1 {{
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700;
    font-size: 2.6rem; line-height: 1.05; margin: 0; color: #FFFFFF; letter-spacing: 0.01em;
}}
.bandeau p {{ color: #C9D1D7; font-size: 1.05rem; margin: 0.5rem 0 1.3rem 0; max-width: 62ch; }}
.marquage {{
    height: 8px; margin: 0 -2rem;
    background: repeating-linear-gradient(90deg, {MARQUAGE} 0 64px, transparent 64px 112px);
}}
.verdict {{
    background: #FFFFFF; border-left: 6px solid {MARQUAGE}; border-radius: 4px;
    padding: 0.9rem 1.2rem; margin: 0.4rem 0 1rem 0; font-size: 1.08rem; color: {ASPHALTE};
}}
.verdict.rien {{ border-left-color: #2E8B57; }}
.legende {{ color: {TEXTE_DOUX}; font-size: 0.92rem; margin-top: 0.2rem; }}
.pastille {{ display: inline-block; width: 0.8em; height: 0.8em; border-radius: 2px; margin: 0 0.3em 0 0.9em; vertical-align: -0.05em; }}
[data-testid="stMetric"] {{ background: #FFFFFF; border-radius: 4px; padding: 0.7rem 1rem; }}
[data-testid="stMetricLabel"] p {{ color: {TEXTE_DOUX}; font-size: 0.9rem; }}
[data-testid="stMetricValue"] {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600; color: {ASPHALTE}; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bandeau">
  <h1>Détecteur d'obstacles routiers</h1>
  <p>Repère, pixel par pixel, les obstacles inattendus sur la route : animaux, cônes, pneus, rochers, charrettes…
  Choisissez une image du jeu de test ou déposez votre propre photo dans le panneau de gauche.</p>
  <div class="marquage"></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Construction des modèles (identique au notebook, étapes 20, 24, 25 et 26)
# ---------------------------------------------------------------------------
def construire(type_modele):
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    def bloc_conv(x, filtres):
        x = layers.Conv2D(filtres, 3, padding="same", activation="relu")(x)
        return layers.Conv2D(filtres, 3, padding="same", activation="relu")(x)

    keras.backend.clear_session()
    entree = keras.Input(shape=(H, W, 3))

    if type_modele in ("unet", "fcn"):
        f = 16
        e1 = bloc_conv(entree, f)
        e2 = bloc_conv(layers.MaxPooling2D()(e1), f * 2)
        e3 = bloc_conv(layers.MaxPooling2D()(e2), f * 4)
        e4 = bloc_conv(layers.MaxPooling2D()(e3), f * 8)
        x = bloc_conv(layers.MaxPooling2D()(e4), f * 16)
        for raccourci, filtres in [(e4, f * 8), (e3, f * 4), (e2, f * 2), (e1, f)]:
            x = layers.UpSampling2D()(x)
            if type_modele == "unet":
                x = layers.Concatenate()([x, raccourci])
            x = bloc_conv(x, filtres)
        sortie = layers.Conv2D(1, 1, activation="sigmoid")(x)
        return keras.Model(entree, sortie, name="modele_1_cnn" if type_modele == "unet" else "modele_3_fcn")

    x = layers.Lambda(lambda t: preprocess_input((t * ECART_TYPE + MOYENNE) * 255.0))(entree)
    resnet = ResNet50(include_top=False, weights=POIDS_RESNET, input_shape=(H, W, 3))

    if type_modele == "unet_resnet":
        noms = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
        encodeur = keras.Model(resnet.input, [resnet.get_layer(n).output for n in noms], name="encodeur_resnet50")
        encodeur.trainable = False
        e1, e2, e3, centre = encodeur(x, training=False)
        d3 = bloc_conv(layers.Concatenate()([layers.UpSampling2D()(centre), e3]), 128)
        d2 = bloc_conv(layers.Concatenate()([layers.UpSampling2D()(d3), e2]), 64)
        d1 = bloc_conv(layers.Concatenate()([layers.UpSampling2D()(d2), e1]), 32)
        d0 = bloc_conv(layers.UpSampling2D()(d1), 16)
        sortie = layers.Conv2D(1, 1, activation="sigmoid")(d0)
        return keras.Model(entree, sortie, name="modele_4_unet_resnet50")

    # DeepLabV3+
    encodeur = keras.Model(resnet.input, [resnet.get_layer("conv2_block3_out").output,
                                          resnet.get_layer("conv4_block6_out").output], name="encodeur_resnet50")
    encodeur.trainable = False
    bas_niveau, profond = encodeur(x, training=False)
    branches = [layers.Conv2D(256, 1, padding="same", activation="relu")(profond)]
    for taux in (6, 12, 18):
        branches.append(layers.Conv2D(256, 3, padding="same", dilation_rate=taux, activation="relu")(profond))
    moy = layers.GlobalAveragePooling2D(keepdims=True)(profond)
    moy = layers.Conv2D(256, 1, activation="relu")(moy)
    branches.append(layers.UpSampling2D(size=(45, 80))(moy))
    aspp = layers.Conv2D(256, 1, activation="relu")(layers.Concatenate()(branches))
    haut = layers.UpSampling2D(size=4, interpolation="bilinear")(aspp)
    detail = layers.Conv2D(48, 1, activation="relu")(bas_niveau)
    d = bloc_conv(layers.Concatenate()([haut, detail]), 256)
    d = layers.Conv2D(1, 1)(d)
    d = layers.UpSampling2D(size=4, interpolation="bilinear")(d)
    sortie = layers.Activation("sigmoid")(d)
    return keras.Model(entree, sortie, name="modele_5_deeplabv3plus")


@st.cache_resource(max_entries=1, show_spinner=False)
def charger_modele(type_modele, fichier):
    modele = construire(type_modele)
    modele.load_weights(os.path.join(DOSSIER_MODELES, fichier + ".weights.h5"))
    return modele


@st.cache_resource(show_spinner=False)
def charger_classifieur():
    """Classifieur animal / autre obstacle (étape 32), s'il a été exporté depuis le notebook."""
    chemin = os.path.join(DOSSIER_MODELES, "classifieur_animal.weights.h5")
    if not os.path.exists(chemin):
        return None
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications.resnet50 import ResNet50
    extracteur = ResNet50(include_top=False, weights=POIDS_RESNET, pooling="avg", input_shape=(224, 224, 3))
    classifieur = keras.Sequential([keras.Input(shape=(2048,)), layers.Dropout(0.5), layers.Dense(1, activation="sigmoid")])
    classifieur.load_weights(chemin)
    return extracteur, classifieur


# ---------------------------------------------------------------------------
# Traitements
# ---------------------------------------------------------------------------
def fr(valeur, decimales=2):
    """Nombre au format français (virgule décimale)."""
    return f"{valeur:.{decimales}f}".replace(".", ",")


def preparer(image):
    """Image quelconque -> tableau 720 × 1280 × 3 (uint8), taille d'entrée des modèles."""
    return np.array(image.convert("RGB").resize((W, H), Image.BILINEAR))


@st.cache_data(max_entries=6, show_spinner=False)
def predire(type_modele, fichier, photo):
    modele = charger_modele(type_modele, fichier)
    x = ((photo.astype(np.float32) / 255.0 - MOYENNE) / ECART_TYPE)[None]
    debut = time.time()
    proba = modele.predict(x, verbose=0)[0, ..., 0]
    return proba, time.time() - debut


def zones_detectees(masque, taille_min=30):
    """Regroupe les pixels « obstacle » voisins en zones ; renvoie les numéros des zones, de la plus grande à la plus petite."""
    etiquettes, n = ndimage.label(masque)
    if n == 0:
        return [], etiquettes
    tailles = ndimage.sum(masque, etiquettes, range(1, n + 1))
    return [int(i) + 1 for i in np.argsort(tailles)[::-1] if tailles[i] >= taille_min], etiquettes


def classer_zones(photo, etiquettes, zones, maxi=5, taille_min=300):
    """Classe chaque grande zone détectée : animal ou autre obstacle (classifieur de l'étape 32)."""
    modeles = charger_classifieur()
    if modeles is None:
        return []
    from tensorflow.keras.applications.resnet50 import preprocess_input
    extracteur, classifieur = modeles
    resultats, recadrages = [], []
    for z in zones[:maxi]:
        zone = etiquettes == z
        if zone.sum() < taille_min:
            continue
        lig, col = np.where(zone)
        mh, mw = int(0.1 * (lig.max() - lig.min() + 1)), int(0.1 * (col.max() - col.min() + 1))
        recadrage = photo[max(0, lig.min() - mh):lig.max() + mh + 1, max(0, col.min() - mw):col.max() + mw + 1]
        recadrages.append(np.array(Image.fromarray(recadrage).resize((224, 224), Image.BILINEAR), dtype=np.float32))
        resultats.append({"part": 100 * zone.mean(), "x": int(col.min()), "y": int(lig.min())})
    if not recadrages:
        return []
    caracteristiques = extracteur.predict(preprocess_input(np.stack(recadrages)), verbose=0)
    for r, p in zip(resultats, classifieur.predict(caracteristiques, verbose=0)[:, 0]):
        r["p_animal"] = float(p)
    return resultats


def numeroter(image, resultats):
    """Écrit le numéro de chaque zone classée sur l'image."""
    img = Image.fromarray(image)
    dessin = ImageDraw.Draw(img)
    try:
        police = ImageFont.load_default(size=28)
    except TypeError:
        police = ImageFont.load_default()
    for k, r in enumerate(resultats, 1):
        x, y = r["x"], max(0, r["y"] - 38)
        boite = dessin.textbbox((x, y), str(k), font=police)
        dessin.rectangle((boite[0] - 7, boite[1] - 5, boite[2] + 7, boite[3] + 5), fill=(35, 41, 46))
        dessin.text((x, y), str(k), fill=(242, 183, 5), font=police)
    return np.array(img)


def teinter(photo, zone, couleur, opacite):
    image = photo.astype(np.float32).copy()
    image[zone] = (1 - opacite) * image[zone] + opacite * np.array(couleur, dtype=np.float32)
    return image.astype(np.uint8)


def contour(masque, epaisseur=2):
    return masque & ~ndimage.binary_erosion(masque, iterations=epaisseur)


def superposition_detection(photo, masque, opacite, masque_ajout=None):
    image = teinter(photo, masque, (242, 183, 5), opacite)
    image[contour(masque)] = (215, 38, 61)
    if masque_ajout is not None:
        image[contour(masque_ajout, 3)] = (255, 255, 255)
    return image


def lister_images():
    if not os.path.isdir(FRAMES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(FRAMES_DIR) if f.endswith(".webp"))


@st.cache_data(show_spinner=False)
def decouper_obstacle(nom):
    """Découpe l'obstacle d'une image du dataset, avec un fond transparent (grâce à son masque)."""
    photo = Image.open(os.path.join(FRAMES_DIR, nom + ".webp")).convert("RGBA")
    masque = np.array(Image.open(os.path.join(FRAMES_DIR, nom + ".labels", "labels_semantic.png"))) == 2
    photo.putalpha(Image.fromarray((masque * 255).astype(np.uint8)))
    lig, col = np.where(masque)
    return np.array(photo.crop((col.min(), lig.min(), col.max() + 1, lig.max() + 1)))


def coller(photo, objet_rgba, pos_x, pos_y, largeur_pct):
    """Colle l'objet sur la photo, centré sur (pos_x, pos_y) en % ; renvoie la photo et le masque de l'objet."""
    objet = Image.fromarray(objet_rgba)
    w = max(4, int(W * largeur_pct / 100))
    h = max(4, int(objet.height * w / objet.width))
    objet = objet.resize((w, h), Image.LANCZOS)
    calque = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    calque.paste(objet, (int(pos_x / 100 * W - w / 2), int(pos_y / 100 * H - h / 2)))
    resultat = Image.alpha_composite(Image.fromarray(photo).convert("RGBA"), calque).convert("RGB")
    return np.array(resultat), np.array(calque)[..., 3] > 127


def superposition_verite(photo, masque, verite, opacite):
    image = teinter(photo, masque & verite, (46, 160, 67), opacite)
    image = teinter(image, masque & ~verite, (215, 38, 61), opacite)
    return teinter(image, ~masque & verite, (37, 99, 235), opacite)


def carte_probabilite(proba):
    return (matplotlib.colormaps["inferno"](proba)[..., :3] * 255).astype(np.uint8)


def metriques(masque, verite):
    vp, fp, fn = (masque & verite).sum(), (masque & ~verite).sum(), (~masque & verite).sum()
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = vp / (vp + fp + fn) if vp + fp + fn else 0.0
    return precision, recall, f1, iou


# ---------------------------------------------------------------------------
# Panneau de gauche
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Image du jeu de test")
    photo, verite, nom_image = None, None, None
    noms = CONFIG.get("images_test") or lister_images()
    if noms:
        nom_image = st.selectbox("Image", noms, label_visibility="collapsed")
        photo = preparer(Image.open(os.path.join(FRAMES_DIR, nom_image + ".webp")))
        chemin_masque = os.path.join(FRAMES_DIR, nom_image + ".labels", "labels_semantic.png")
        if os.path.exists(chemin_masque):
            verite = np.array(Image.open(chemin_masque).resize((W, H), Image.NEAREST)) == 2
    else:
        st.warning("Dossier du dataset introuvable : placez data/RoadAnomaly à côté de app.py, "
                   "ou déposez une nouvelle image ci-dessous.")

    st.subheader("Déposer une nouvelle image")
    fichier_image = st.file_uploader("Photo de route (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"],
                                     help="Une photo déposée est analysée à la place de l'image du jeu de test. "
                                          "Retirez-la (croix) pour revenir au jeu de test.")
    if fichier_image is not None:
        nom_image = fichier_image.name
        photo = preparer(Image.open(fichier_image))
        verite = None
        st.caption("Analyse de la photo déposée : le type de chaque obstacle détecté est indiqué en haut de la page.")

    objet_rgba = None
    with st.expander("Option : coller un objet sur la photo"):
        ajouter = st.checkbox("Coller un objet sur la photo",
                              help="Place un objet où vous voulez sur la route, pour tester si le modèle le détecte.")
        if ajouter:
            origine = st.radio("Objet", ["Un obstacle du dataset", "Mon image"], horizontal=True)
            if origine == "Un obstacle du dataset":
                noms_tous = lister_images()
                if noms_tous:
                    nom_objet = st.selectbox("Obstacle à copier", noms_tous)
                    objet_rgba = decouper_obstacle(nom_objet)
                else:
                    st.warning("Dataset introuvable : importez plutôt votre propre image d'objet.")
            else:
                fichier_objet = st.file_uploader("Image de l'objet (PNG à fond transparent conseillé)",
                                                 type=["png", "jpg", "jpeg", "webp"], key="objet")
                if fichier_objet is not None:
                    objet_rgba = np.array(Image.open(fichier_objet).convert("RGBA"))
            pos_x = st.slider("Position horizontale (%)", 0, 100, 50)
            pos_y = st.slider("Position verticale (%)", 0, 100, 70)
            taille = st.slider("Largeur de l'objet (% de la largeur de l'image)", 1, 60, 15,
                               help="Petit = obstacle lointain, grand = obstacle proche.")

    st.subheader("Modèle")
    nom_modele = st.selectbox("Modèle", list(MODELES), label_visibility="collapsed")
    type_modele, fichier_poids, scores = MODELES[nom_modele]
    st.caption(f"Sur le test : F1 {fr(scores['F1'])}, recall {fr(scores['Recall'])}, "
               f"precision {fr(scores['Precision'])}, fond signalé à tort {fr(scores['FPR'])} %")

    st.subheader("Réglages")
    seuil = st.slider("Seuil de détection", 0.10, 0.90, 0.50, 0.05,
                      help="Probabilité à partir de laquelle un pixel est déclaré obstacle. "
                           "Plus bas : moins d'obstacles ratés, plus de fausses alertes.")
    opacite = st.slider("Opacité de la superposition", 0.2, 0.9, 0.55, 0.05)

# ---------------------------------------------------------------------------
# Contenu principal
# ---------------------------------------------------------------------------
if photo is None:
    st.info("Choisissez une image du jeu de test ou déposez une photo de route dans le panneau de gauche.")
    st.stop()

if not os.path.exists(os.path.join(DOSSIER_MODELES, fichier_poids + ".weights.h5")):
    st.error(f"Poids introuvables : {fichier_poids}.weights.h5. Placez le dossier RoadAnomaly_modeles "
             "(téléchargé depuis Google Drive) à côté de app.py.")
    st.stop()

masque_ajout = None
if objet_rgba is not None:
    photo, masque_ajout = coller(photo, objet_rgba, pos_x, pos_y, taille)
    if verite is not None:
        verite = verite | masque_ajout

with st.spinner("Analyse de l'image…"):
    proba, duree = predire(type_modele, fichier_poids, photo)

masque = proba >= seuil
zones, etiquettes = zones_detectees(masque)
nb_zones = len(zones)
part = 100 * masque.mean()
with st.spinner("Identification du type de chaque obstacle…"):
    classees = classer_zones(photo, etiquettes, zones)

if nb_zones:
    texte = f"<b>{nb_zones} zone{'s' if nb_zones > 1 else ''} d'obstacle détectée{'s' if nb_zones > 1 else ''}</b>, " \
            f"soit {fr(part)} % de l'image."
    for k, r in enumerate(classees, 1):
        genre = "animal" if r["p_animal"] >= 0.5 else "autre obstacle (objet, rocher, véhicule…)"
        confiance = r["p_animal"] if r["p_animal"] >= 0.5 else 1 - r["p_animal"]
        texte += f"<br>Obstacle {k} : <b>{genre}</b>, confiance {100 * confiance:.0f} % ({fr(r['part'])} % de l'image)"
    if classees and nb_zones > len(classees):
        texte += f"<br><span style='color:{TEXTE_DOUX}'>Les {len(classees)} plus grandes zones sont classées ; les autres sont trop petites.</span>"
else:
    texte = "<b>Aucun obstacle détecté</b> au seuil choisi. Baissez le seuil pour repérer des zones moins sûres."

if masque_ajout is not None and masque_ajout.any():
    couverture = (masque & masque_ajout).sum() / masque_ajout.sum()
    statut = "<b>détecté</b>" if couverture >= 0.5 else "<b>non détecté</b>"
    texte += (f"<br>Objet ajouté ({fr(100 * masque_ajout.mean())} % de l'image) : {statut}, "
              f"{100 * couverture:.0f} % de sa surface reconnue comme obstacle (détecté à partir de 50 %, comme à l'étape 29).")

st.markdown(f'<div class="verdict{"" if nb_zones else " rien"}">{texte}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Zones détectées", nb_zones)
c2.metric("Part de l'image signalée", f"{fr(part)} %")
c3.metric("Confiance maximale", f"{100 * proba.max():.0f} %")
c4.metric("Temps d'analyse", f"{fr(duree, 1)} s")

onglets = st.tabs(["Détection", "Obstacle / fond", "Carte de probabilité", "Comparaison avec le vrai masque", "Les 5 modèles"])

with onglets[0]:
    g, d = st.columns(2)
    g.image(photo, caption=f"Photo : {nom_image}")
    d.image(numeroter(superposition_detection(photo, masque, opacite, masque_ajout), classees),
            caption="Obstacles détectés (numérotés comme dans le message du haut)")
    st.markdown(f'<div class="legende"><span class="pastille" style="background:{MARQUAGE}"></span>zone détectée'
                '<span class="pastille" style="background:#D7263D"></span>contour de la zone'
                + ('<span class="pastille" style="background:#FFFFFF;border:1px solid #23292E"></span>objet ajouté'
                   if masque_ajout is not None else '') + '</div>',
                unsafe_allow_html=True)

with onglets[1]:
    st.image(np.where(masque[..., None], np.array([242, 183, 5], np.uint8), np.array([35, 41, 46], np.uint8)),
             caption="Séparation obstacle / fond : jaune = obstacle, gris foncé = fond")

with onglets[2]:
    st.image(carte_probabilite(proba),
             caption="Probabilité d'obstacle pour chaque pixel : noir = sûrement du fond, jaune clair = sûrement un obstacle")

with onglets[3]:
    if verite is None:
        st.info("La comparaison n'est possible que pour les images du jeu de test, qui ont un masque annoté.")
    else:
        precision, recall, f1, iou = metriques(masque, verite)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", fr(precision))
        m2.metric("Recall", fr(recall))
        m3.metric("F1 / Dice", fr(f1))
        m4.metric("IoU", fr(iou))
        st.image(superposition_verite(photo, masque, verite, opacite))
        st.markdown('<div class="legende"><span class="pastille" style="background:#2EA043"></span>obstacle bien détecté'
                    '<span class="pastille" style="background:#D7263D"></span>fausse alerte'
                    '<span class="pastille" style="background:#2563EB"></span>obstacle raté</div>',
                    unsafe_allow_html=True)

with onglets[4]:
    st.markdown("Scores obtenus par chaque modèle sur les 10 images de test du projet (seuil 0,5).")
    st.dataframe(
        {"Modèle": list(MODELES),
         "F1": [s["F1"] for _, _, s in MODELES.values()],
         "Recall": [s["Recall"] for _, _, s in MODELES.values()],
         "Precision": [s["Precision"] for _, _, s in MODELES.values()],
         "IoU": [s["IoU"] for _, _, s in MODELES.values()],
         "Fond signalé à tort (%)": [s["FPR"] for _, _, s in MODELES.values()]},
        hide_index=True)
    st.markdown("- **DeepLabV3+ avec ResNet50** est retenu : meilleur F1 et meilleur recall, critère prioritaire "
                "pour un véhicule (rater un obstacle est plus grave qu'une fausse alerte).\n"
                "- **Limite connue** : les très petits obstacles lointains (moins de 0,1 % de l'image) sont souvent manqués.\n"
                "- Les photos importées sont redimensionnées en 1280 × 720, la taille d'entraînement des modèles.")
