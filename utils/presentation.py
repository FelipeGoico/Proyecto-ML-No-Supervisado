import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Iterable
from IPython.display import display
from pathlib import Path


DEFAULT_WIDTH = 50


def show_section(title: str, emoji: Optional[str] = None, width: int = DEFAULT_WIDTH) -> None:
    """Imprime un título en MAYÚSCULAS con separador estándar y un salto de línea.
    - title: texto del título (se convertirá a MAYÚSCULAS)
    - emoji: emoji opcional que se antepone al título
    - width: largo del separador (por defecto 50)
    """
    line = title.upper()
    if emoji:
        line = f"{emoji} {line}"
    print(line)
    print("=" * width)
    print()


def show_subsection(text: str, emoji: Optional[str] = None) -> None:
    """Imprime un subtítulo en MAYÚSCULAS con un salto de línea previo."""
    line = text.upper()
    if emoji:
        line = f"{emoji} {line}"
    print()
    print(line)


def show_list(items: Iterable, bullet: str = "-") -> None:
    """Imprime una lista simple con viñetas."""
    for it in items:
        print(f"{bullet} {it}")


def print_kv(key: str, value, bullet: str = "-") -> None:
    """Imprime una línea tipo clave: valor con viñeta opcional."""
    print(f"{bullet} {key}: {value}")


def blank(n: int = 1) -> None:
    """Imprime n saltos de línea."""
    for _ in range(n):
        print()


# ================================================================================
# FUNCIONES DE ANÁLISIS DE DATASETS ESPECÍFICOS
# ================================================================================

def create_datasets_summary_table():
    """Crea y muestra la tabla resumen estilizada de los 10 datasets."""
    datasets_info = {
        "No.": list(range(1, 11)),
        "Dataset": [
            "Dairy Goods Sales Dataset", "Animal Sounds Dataset",
            "Customer Personality Analysis", "News Category Dataset",
            "Global Coffee Health Dataset", "Credit Card Customer",
            "Sleep Health and Lifestyle", "Wholesale Customers Dataset",
            "PlantVillage Dataset", "Milk Quality Dataset"
        ],
        "Dominio": [
            "Comercio/Ventas", "Audio/Sonidos", "Comportamiento del Consumidor",
            "Texto/NLP", "Salud Global", "Servicios Financieros",
            "Salud y Bienestar", "Comercio B2B", "Visión por Computador",
            "Calidad Alimentaria"
        ],
        "Tipo_de_Datos": [
            "Transaccional", "Audio (WAV)", "Demográfico/Comportamental",
            "Texto/JSON", "Numérico/Geográfico", "Financiero/Numérico",
            "Médico/Estilo de vida", "Ventas por categoría",
            "Imágenes RGB", "Fisicoquímico/Numérico"
        ],
        "Formato_Archivo": [
            "CSV", "WAV/Audio", "CSV", "JSON", "CSV",
            "CSV", "CSV", "CSV", "JPG/PNG", "CSV"
        ],
        "URL_Origen": [
            "https://www.kaggle.com/datasets/suraj520/dairy-goods-sales-dataset",
            "https://github.com/YashNita/Animal-Sound-Dataset",
            "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis",
            "https://www.kaggle.com/datasets/rmisra/news-category-dataset",
            "https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset",
            "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers",
            "https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset",
            "https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set",
            "https://www.kaggle.com/datasets/plantvillage-dataset",
            "https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"
        ],
        "Aplicaciones_ML_No_Supervisado": [
            "Segmentación clientes, Patrones compra",
            "Clustering sonidos, Clasificación no supervisada",
            "Segmentación personalidad, Perfiles cliente",
            "Topic modeling, Clustering temas",
            "Clustering geográfico, Patrones consumo",
            "Detección fraudes, Clustering crediticio",
            "Patrones sueño, Clustering estilos vida",
            "Segmentación B2B, Patrones compra",
            "Clustering visual, Detección anomalías",
            "Control calidad, Detección anomalías"
        ]
    }

    df_summary = pd.DataFrame(datasets_info)

    # Estilo visual con pandas Styler
    styled_df = (
        df_summary.style
        .set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#4C72B0"), ("color", "white"), ("font-weight", "bold")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
            {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "white")]}
        ])
        .set_properties(**{
            "text-align": "left",
            "border": "1px solid #ddd",
            "color": "#222222"
        })
        .hide(axis="index")
    )

    show_section("Tabla resumen de los 10 datasets", emoji="📋")
    display(styled_df)

    # Resumen general
    blank()
    print_kv("Total de datasets analizados", len(df_summary))
    print_kv("Dominios cubiertos", df_summary["Dominio"].nunique())
    print_kv("Tipos de formato", ", ".join(df_summary["Formato_Archivo"].unique()))

    return df_summary


# -------------------- Función combinada de Dairy Dataset -------------------- #
def analyze_dairy_dataset(df: pd.DataFrame) -> None:
    """Muestra descripción detallada y realiza EDA visual del Dairy Goods Sales Dataset."""
    df.columns = df.columns.str.strip()  # limpiar espacios

    # ===== Carga y descripción ===== #
    show_subsection("Carga y descripción del conjunto de datos", emoji="✅")
    print_kv("Filas × Columnas", f"{df.shape[0]:,} × {df.shape[1]}")
    print_kv("Uso de memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    show_subsection("Columnas disponibles", emoji="📋")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        unique_vals = df[col].nunique() if df[col].dtype in ['object', 'int64'] else 'continua'
        print(f"  {i:2d}. {col:30s} ({dtype:8s}) - {unique_vals} valores únicos")

    show_subsection("Estadísticas clave", emoji="📈")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print_kv("Variables numéricas", len(numeric_cols))
        for col in numeric_cols[:5]:  # primeras 5 para no saturar
            min_val, max_val = df[col].min(), df[col].max()
            mean_val = df[col].mean()
            print(f"  * {col}: min={min_val}, max={max_val}, promedio={mean_val:.2f}")

    null_counts = df.isnull().sum()
    completeness = (1 - null_counts.sum() / (df.shape[0] * df.shape[1])) * 100
    print_kv("Completitud", f"{completeness:.1f}%")

    # ===== EDA visual ===== #
    show_subsection("EDA visual", emoji="📊")
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16,10))

    # 1️⃣ Ventas totales por producto
    sns.barplot(
        x='Product Name', y='Quantity Sold (liters/kg)', 
        data=df, estimator=sum, ax=axes[0,0], palette='viridis'
    )
    axes[0,0].set_title("Ventas totales por Producto")
    axes[0,0].tick_params(axis='x', rotation=45)

    # 2️⃣ Distribución de ventas por canal de venta
    sns.boxplot(
        x='Sales Channel', y='Quantity Sold (liters/kg)',
        data=df, ax=axes[0,1], palette='Set2'
    )
    axes[0,1].set_title("Distribución de Ventas por Canal")
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3️⃣ Matriz de correlación numérica
    numeric_cols = ['Quantity (liters/kg)', 'Price per Unit', 'Total Value', 'Shelf Life (days)', 'Quantity in Stock (liters/kg)']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1,0], fmt='.2f', linewidths=0.5)
    axes[1,0].set_title("Matriz de Correlación")

    # 4️⃣ Tendencia de ventas mensuales (corregida)
    try:
        df_temp = df.copy()  # No modificar el DataFrame original
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp['Month_Str'] = df_temp['Date'].dt.strftime('%Y-%m')  # Formato string para evitar problemas con Period
        monthly_sales = df_temp.groupby('Month_Str')['Quantity Sold (liters/kg)'].sum().reset_index()
        
        # Convertir de vuelta a datetime para el gráfico
        monthly_sales['Month_Date'] = pd.to_datetime(monthly_sales['Month_Str'] + '-01')
        
        axes[1,1].plot(monthly_sales['Month_Date'], monthly_sales['Quantity Sold (liters/kg)'], 
                      marker='o', color='b', linewidth=2)
        axes[1,1].set_title("Ventas Mensuales Totales")
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
    except Exception as e:
        # Si hay error en las fechas, mostrar un gráfico alternativo
        axes[1,1].text(0.5, 0.5, f'Gráfico temporal no disponible\n(Error: {str(e)[:50]}...)', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title("Ventas Mensuales (No disponible)")

    plt.tight_layout()
    plt.show()



    # ======================================
# NUEVA FUNCIÓN: ANALYZE ANIMAL SOUNDS DATASET
# ======================================

def analyze_animal_sounds_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        print("⚠️ Dataset no encontrado, disponible en:")
        print("https://github.com/YashNita/Animal-Sound-Dataset")
        return

    show_section("Animal Sounds Dataset", emoji="🐾")

    # Traducción de nombres al castellano
    animal_names = {
        'Aslan': 'León',
        'Esek': 'Burro', 
        'Inek': 'Vaca',
        'Kedi-Part1': 'Gato (Parte 1)',
        'Kedi-Part2': 'Gato (Parte 2)',
        'Kopek-Part1': 'Perro (Parte 1)',
        'Kopek-Part2': 'Perro (Parte 2)',
        'Koyun': 'Oveja',
        'Kurbaga': 'Rana',
        'Kus-Part1': 'Pájaro (Parte 1)',
        'Kus-Part2': 'Pájaro (Parte 2)',
        'Maymun': 'Mono',
        'Tavuk': 'Pollo'
    }

    # 1️⃣ Descripción general y estructura
    folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    show_subsection("Descripción del dataset", emoji="📊")
    print_kv("Número de categorías de animales", len(folders))

    total_files = 0
    category_info = {}
    audio_samples = {}  # Para almacenar ejemplos de audio por animal
    
    for folder in folders:
        audio_files = list(folder.glob("*.wav"))
        total_files += len(audio_files)
        category_info[folder.name] = len(audio_files)
        # Guardar el primer archivo como ejemplo para reproducción
        if audio_files:
            audio_samples[folder.name] = audio_files[0]

    print_kv("Total de archivos de audio", f"{total_files:,}")

    show_subsection("Distribución por especies", emoji="📦")
    for animal, count in sorted(category_info.items()):
        animal_castellano = animal_names.get(animal, animal)
        print(f"- {animal_castellano}: {count} archivos")

    # 2️⃣ Reproducción interactiva de sonidos de ejemplo
    show_subsection("🎵 Ejemplos de Sonidos por Especie", emoji="🔊")
    print("📌 Especies disponibles en el dataset:")
    print()
    
    # Mostrar opciones disponibles en castellano
    available_animals = list(audio_samples.keys())
    for i, animal in enumerate(available_animals, 1):
        animal_castellano = animal_names.get(animal, animal)
        print(f"   {i}. {animal_castellano} ({animal})")
    
    print(f"\n💡 Para reproducir un sonido, ejecuta en la siguiente celda:")
    print(f"   # Ejemplo: Reproducir sonido del animal en posición 1")
    print(f"   from IPython.display import Audio, display")
    print(f"   from pathlib import Path")
    print(f"   ")
    print(f"   # Opciones disponibles:")
    for i, animal in enumerate(available_animals, 1):
        if animal in audio_samples:
            file_path = audio_samples[animal]
            animal_castellano = animal_names.get(animal, animal)
            print(f"   # {i}. {animal_castellano}: display(Audio('{file_path}'))")
    
    print(f"\n🎧 Código de ejemplo para reproducir cualquier sonido:")
    print(f"   animal_elegido = '{available_animals[0] if available_animals else 'Aslan'}'")
    print(f"   audio_path = Path('Animal Sound Dataset') / animal_elegido")
    print(f"   if audio_path.exists():")
    print(f"       audio_files = list(audio_path.glob('*.wav'))")
    print(f"       if audio_files:")
    print(f"           display(Audio(audio_files[0]))")

    # 3️⃣ Características técnicas (muestra)
    if folders:
        sample_folder = folders[0]
        sample_files = list(sample_folder.glob("*.wav"))
        if sample_files:
            sample_file = sample_files[0]
            file_size = sample_file.stat().st_size
            show_subsection("Características técnicas (muestra)", emoji="🔧")
            show_list([
                "Formato: WAV (Waveform Audio File Format)",
                f"Tamaño aproximado del archivo de muestra: ~{file_size/1024:.1f} KB",
                "Tipo de datos: Señales de audio digitales",
                "Dominio: Bioacústica y reconocimiento de especies",
            ])
    else:
        print("⚠️ No se encontraron archivos de audio en las carpetas.")

    # 4️⃣ Posibles aplicaciones no supervisadas
    show_subsection("Líneas de análisis no supervisado", emoji="🎯")
    show_list([
        "Clustering espectral para identificar grupos acústicos",
        "Detección de anomalías en vocalizaciones",
        "Análisis de componentes principales en características MFCC",
        "Segmentación automática de comportamientos vocales",
    ])

    # 5️⃣ EDA visual compacto con análisis espectral
    show_subsection("EDA visual", emoji="📊")
    fig, axes = plt.subplots(2, 2, figsize=(16,10))
    axes = axes.flatten()

    # Distribución de cantidad de audios por categoría (barplot)
    try:
        # Crear listas con nombres en castellano para el gráfico
        animals_castellano = [animal_names.get(animal, animal) for animal in category_info.keys()]
        counts = list(category_info.values())
        
        sns.barplot(
            x=animals_castellano, y=counts,
            ax=axes[0], palette='husl'
        )
        axes[0].set_title("Cantidad de archivos por especie")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylabel("Cantidad de audios")
    except:
        axes[0].text(0.5,0.5,"No disponible", ha='center')

    # Ejemplo de espectrograma de un audio
    try:
        if 'sample_file' in locals():
            y, sr = librosa.load(sample_file, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
            axes[1].set_title(f"Espectrograma: {sample_file.name}")
        else:
            axes[1].text(0.5,0.5,"No disponible", ha='center')
    except:
        axes[1].text(0.5,0.5,"No disponible", ha='center')

    # Forma de onda del audio
    try:
        if 'y' in locals() and 'sr' in locals():
            time_axis = np.linspace(0, len(y)/sr, len(y))
            axes[2].plot(time_axis, y, alpha=0.7)
            axes[2].set_title(f"Forma de onda: {sample_file.name}")
            axes[2].set_xlabel("Tiempo (s)")
            axes[2].set_ylabel("Amplitud")
        else:
            axes[2].text(0.5,0.5,"No disponible", ha='center')
    except:
        axes[2].text(0.5,0.5,"No disponible", ha='center')

    # Análisis de frecuencias (FFT)
    try:
        if 'y' in locals() and 'sr' in locals():
            fft = np.fft.fft(y)
            freq = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            # Solo mostrar frecuencias positivas
            pos_mask = freq >= 0
            axes[3].plot(freq[pos_mask][:len(freq)//8], magnitude[pos_mask][:len(freq)//8])
            axes[3].set_title("Espectro de Frecuencias (FFT)")
            axes[3].set_xlabel("Frecuencia (Hz)")
            axes[3].set_ylabel("Magnitud")
        else:
            axes[3].text(0.5,0.5,"No disponible", ha='center')
    except:
        axes[3].text(0.5,0.5,"No disponible", ha='center')

    plt.tight_layout()
    plt.show()

    print()
    print("✅ Análisis del Animal Sounds Dataset completado exitosamente")
    print("🎵 ¡Usa el código de ejemplo de arriba para reproducir sonidos de los animales!")


def play_animal_sounds_interactive(dataset_path: Path) -> None:
    """
    Función interactiva para reproducir sonidos de animales del dataset.
    Reproduce automáticamente un sonido aleatorio y muestra la forma de onda.
    """
    from IPython.display import Audio, display
    import random
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not dataset_path.exists():
        print("⚠️ Dataset 'Animal Sound Dataset' no encontrado")
        print("📁 Verifica que la carpeta existe en el directorio actual")
        return

    show_section("Reproducción de Sonidos de Animales", emoji="🎵")
    
    # Traducción de nombres al castellano
    animal_names = {
        'Aslan': 'León',
        'Esek': 'Burro', 
        'Inek': 'Vaca',
        'Kedi-Part1': 'Gato (Parte 1)',
        'Kedi-Part2': 'Gato (Parte 2)',
        'Kopek-Part1': 'Perro (Parte 1)',
        'Kopek-Part2': 'Perro (Parte 2)',
        'Koyun': 'Oveja',
        'Kurbaga': 'Rana',
        'Kus-Part1': 'Pájaro (Parte 1)',
        'Kus-Part2': 'Pájaro (Parte 2)',
        'Maymun': 'Mono',
        'Tavuk': 'Pollo'
    }
    
    # Obtener todas las especies disponibles
    folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    if not folders:
        print("⚠️ No se encontraron carpetas de animales")
        return
    
    # Reproducir un sonido aleatorio automáticamente
    random_folder = random.choice(folders)
    audio_files = list(random_folder.glob("*.wav"))
    
    if audio_files:
        random_file = random.choice(audio_files)
        animal_castellano = animal_names.get(random_folder.name, random_folder.name)
        
        print("🎵 **REPRODUCIENDO SONIDO ALEATORIO**")
        print("=" * 50)
        print(f"🐾 **Animal**: {animal_castellano}")
        print(f"📁 **Archivo**: {random_file.name}")
        print("=" * 50)
        
        try:
            # Cargar y analizar el audio
            y, sr = librosa.load(random_file, sr=None)
            duration = len(y) / sr
            
            print(f"📊 **Características del audio**:")
            print(f"   • Duración: {duration:.2f} segundos")
            print(f"   • Frecuencia de muestreo: {sr:,} Hz")
            print(f"   • Número de muestras: {len(y):,}")
            
            blank()
            
            # Crear visualización de la forma de onda
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            time = np.linspace(0, duration, len(y))
            ax.plot(time, y, color='steelblue', alpha=0.8)
            ax.set_title(f'Forma de Onda - {animal_castellano}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tiempo (segundos)')
            ax.set_ylabel('Amplitud')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, duration)
            plt.tight_layout()
            plt.show()
            
            blank()
            print("🔊 **Reproduciendo audio...**")
            display(Audio(random_file))
            blank()
            print("✅ **Audio reproducido exitosamente**")
            
        except Exception as e:
            print(f"⚠️ Error al procesar el audio: {e}")
            print("🔊 **Reproduciendo audio...**")
            display(Audio(random_file))
            print("✅ **Audio reproducido exitosamente**")
    else:
        print("⚠️ No se encontraron archivos de audio en las carpetas")


# ======================================
# NUEVA FUNCIÓN: ANALYZE CUSTOMER PERSONALITY DATASET
# ======================================

def analyze_customer_personality_dataset(df: pd.DataFrame) -> None:
    """Muestra descripción detallada y realiza EDA visual del Customer Personality Analysis Dataset."""
    
    # ===== Carga y descripción ===== #
    show_subsection("Carga y descripción del conjunto de datos", emoji="✅")
    print_kv("Filas × Columnas", f"{df.shape[0]:,} × {df.shape[1]}")
    print_kv("Uso de memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    show_subsection("Columnas disponibles", emoji="📋")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        unique_vals = df[col].nunique() if df[col].dtype in ['object', 'int64'] else 'continua'
        print(f"  {i:2d}. {col:30s} ({dtype:8s}) - {unique_vals} valores únicos")

    # ===== Análisis demográfico ===== #
    show_subsection("Análisis demográfico", emoji="👥")
    
    # Crear variables derivadas para el análisis
    df_analysis = df.copy()
    current_year = 2024
    df_analysis['Age'] = current_year - df_analysis['Year_Birth']
    
    # Variables de gasto total
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df_analysis['Total_Spending'] = df_analysis[spending_cols].sum(axis=1)
    
    # Variables de compras por canal
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df_analysis['Total_Purchases'] = df_analysis[purchase_cols].sum(axis=1)
    
    print_kv("Edad promedio", f"{df_analysis['Age'].mean():.1f} años")
    print_kv("Rango de edad", f"{df_analysis['Age'].min()} - {df_analysis['Age'].max()} años")
    print_kv("Gasto promedio total", f"${df_analysis['Total_Spending'].mean():.2f}")
    print_kv("Ingreso promedio", f"${df_analysis['Income'].mean():.2f}")
    
    # Estadísticas por categorías
    show_subsection("Distribución por categorías", emoji="📊")
    print_kv("Niveles educativos", df_analysis['Education'].value_counts().to_dict())
    print_kv("Estados civiles", df_analysis['Marital_Status'].value_counts().to_dict())
    
    # Valores nulos y completitud
    null_counts = df_analysis.isnull().sum()
    completeness = (1 - null_counts.sum() / (df_analysis.shape[0] * df_analysis.shape[1])) * 100
    print_kv("Completitud", f"{completeness:.1f}%")
    if null_counts.sum() > 0:
        print_kv("Columnas con valores nulos", null_counts[null_counts > 0].to_dict())

    # ===== EDA visual ===== #
    show_subsection("EDA visual", emoji="📈")
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16,10))

    # 1️⃣ Distribución de edad vs ingreso
    scatter = axes[0,0].scatter(df_analysis['Age'], df_analysis['Income'], 
                               c=df_analysis['Total_Spending'], cmap='viridis', alpha=0.6)
    axes[0,0].set_xlabel('Edad')
    axes[0,0].set_ylabel('Ingreso')
    axes[0,0].set_title('Edad vs Ingreso (coloreado por Gasto Total)')
    plt.colorbar(scatter, ax=axes[0,0])

    # 2️⃣ Gasto por categoría de producto
    spending_means = df_analysis[spending_cols].mean()
    sns.barplot(x=spending_means.index, y=spending_means.values, ax=axes[0,1], palette='Set2')
    axes[0,1].set_title('Gasto Promedio por Categoría de Producto')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylabel('Gasto Promedio ($)')

    # 3️⃣ Matriz de correlación de variables de gasto
    corr_data = df_analysis[spending_cols + ['Income', 'Total_Spending']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=axes[1,0], 
                fmt='.2f', linewidths=0.5, square=True)
    axes[1,0].set_title('Correlación entre Variables de Gasto e Ingreso')

    # 4️⃣ Distribución de canales de compra
    purchase_means = df_analysis[purchase_cols].mean()
    sns.barplot(x=purchase_means.index, y=purchase_means.values, ax=axes[1,1], palette='husl')
    axes[1,1].set_title('Compras Promedio por Canal')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_ylabel('Compras Promedio')

    plt.tight_layout()
    plt.show()

    # ===== Análisis de segmentación potencial ===== #
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Segmentación por patrones de gasto y preferencias de producto",
        "Clustering por canales de compra preferidos (online vs offline)",
        "Identificación de perfiles demográficos por comportamiento",
        "Detección de clientes de alto valor (outliers en gasto)",
        "Análisis de respuesta a campañas de marketing",
    ])

    show_subsection("Técnicas recomendadas", emoji="🔬")
    show_list([
        "K-Means para segmentación clásica de clientes",
        "DBSCAN para identificar clientes atípicos de alto/bajo valor",
        "PCA para reducir dimensionalidad de variables de gasto",
        "Gaussian Mixture Models para perfiles probabilísticos",
    ])

    print()
    print("✅ Análisis del Customer Personality Analysis Dataset completado exitosamente")


# ======================================
# FUNCIONES DE ANÁLISIS TEÓRICO (DATASETS SIN DATOS LOCALES)
# ======================================

def analyze_news_category_dataset() -> None:
    """Análisis teórico del News Category Dataset basado en fuente web."""
    
    show_subsection("Análisis basado en fuente web", emoji="🌐")
    print_kv("URL de origen", "https://www.kaggle.com/datasets/rmisra/news-category-dataset")
    print_kv("Estado del dataset", "No disponible localmente - Análisis teórico")
    
    show_subsection("Características esperadas del dataset", emoji="📊")
    print_kv("Tamaño estimado", "~210,000 artículos de noticias (HuffPost)")
    print_kv("Formato", "JSON")
    print_kv("Período temporal", "2012-2022")
    print_kv("Variables principales", "headline, short_description, category, authors, date")
    
    show_subsection("Estructura de datos esperada", emoji="📋")
    expected_columns = [
        "headline (título del artículo)",
        "short_description (descripción breve)",
        "category (categoría de noticia)",
        "authors (autores)",
        "date (fecha de publicación)",
        "link (enlace web)"
    ]
    for i, col in enumerate(expected_columns, 1):
        print(f"  {i}. {col}")
    
    show_subsection("Análisis de dominio", emoji="📰")
    print_kv("Diversidad temática", "Política, entretenimiento, deportes, tecnología, mundo, etc.")
    print_kv("Longitud promedio titular", "~50 caracteres")
    print_kv("Longitud promedio descripción", "~200 caracteres")
    print_kv("Idioma", "Inglés")
    
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Topic Modeling con LDA para descubrir temas emergentes",
        "Clustering semántico basado en embeddings (Word2Vec/BERT)",
        "Detección automática de tendencias noticiosas",
        "Análisis de sentimientos no supervisado",
        "Segmentación de audiencias por preferencias temáticas"
    ])
    
    show_subsection("Técnicas aplicables", emoji="🔬")
    show_list([
        "TF-IDF + K-Means para clustering semántico",
        "NMF (Non-negative Matrix Factorization) para topic modeling",
        "DBSCAN para detección de noticias atípicas",
        "t-SNE/UMAP para visualización de clusters textuales"
    ])
    
    print()
    print("✅ Análisis teórico del News Category Dataset completado")


def analyze_coffee_health_dataset() -> None:
    """Análisis teórico del Global Coffee Health Dataset basado en fuente web."""
    
    show_subsection("Análisis basado en fuente web", emoji="🌐")
    print_kv("URL de origen", "https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset")
    print_kv("Estado del dataset", "No disponible localmente - Análisis teórico")
    
    show_subsection("Características esperadas del dataset", emoji="📊")
    print_kv("Alcance geográfico", "~180+ países/regiones globales")
    print_kv("Período temporal", "Datos longitudinales 2000-2020")
    print_kv("Tipo de datos", "Numérico/geográfico con indicadores de salud")
    print_kv("Variables principales", "consumo per cápita, esperanza de vida, indicadores socioeconómicos")
    
    show_subsection("Variables esperadas por categoría", emoji="☕")
    health_vars = [
        "Coffee_Consumption_Per_Capita (kg/año)",
        "Life_Expectancy (años)",
        "Heart_Disease_Rate (%)",
        "Diabetes_Prevalence (%)",
        "GDP_Per_Capita (USD)",
        "Healthcare_Index (0-100)"
    ]
    for i, var in enumerate(health_vars, 1):
        print(f"  {i}. {var}")
    
    show_subsection("Cobertura regional esperada", emoji="🗺️")
    print_kv("Regiones incluidas", "Europa, América, Asia, África, Oceanía")
    print_kv("Variabilidad cultural", "Diferentes patrones de consumo por región")
    print_kv("Datos urbanos vs rurales", "Segmentación por tipo de población")
    
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Clustering geográfico por patrones salud-consumo",
        "Detección de países outliers con patrones únicos",
        "PCA para identificar factores principales de salud global",
        "Segmentación epidemiológica por perfiles regionales",
        "Análisis de correlaciones no lineales café-salud"
    ])
    
    show_subsection("Técnicas recomendadas", emoji="🔬")
    show_list([
        "K-Means para agrupación geográfica de países",
        "DBSCAN para detectar países con patrones atípicos",
        "Hierarchical clustering para taxonomía regional",
        "Gaussian Mixture Models para perfiles probabilísticos"
    ])
    
    print()
    print("✅ Análisis teórico del Global Coffee Health Dataset completado")


def analyze_credit_card_dataset() -> None:
    """Análisis teórico del Credit Card Customer Dataset basado en fuente web."""
    
    show_subsection("Análisis basado en fuente web", emoji="🌐")
    print_kv("URL de origen", "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers")
    print_kv("Estado del dataset", "No disponible localmente - Análisis teórico")
    
    show_subsection("Características esperadas del dataset", emoji="📊")
    print_kv("Tamaño estimado", "~10,000 registros de clientes")
    print_kv("Variables esperadas", "20+ columnas financieras y demográficas")
    print_kv("Dominio", "Servicios financieros - Gestión de riesgo crediticio")
    
    show_subsection("Variables financieras clave esperadas", emoji="💳")
    financial_vars = [
        "Credit_Limit (límite de crédito autorizado)",
        "Total_Revolving_Bal (balance rotativo actual)",
        "Total_Trans_Amt (monto total de transacciones)",
        "Total_Trans_Ct (cantidad de transacciones)",
        "Avg_Utilization_Ratio (ratio de utilización promedio)",
        "Customer_Age (edad del cliente)",
        "Income_Category (categoría de ingresos)",
        "Card_Category (tipo de tarjeta)"
    ]
    for i, var in enumerate(financial_vars, 1):
        print(f"  {i}. {var}")
    
    show_subsection("Perfiles de clientes esperados", emoji="👥")
    print_kv("Segmentos demográficos", "Edad, género, estado civil, educación")
    print_kv("Comportamiento crediticio", "Usuarios activos vs inactivos")
    print_kv("Patrones de gasto", "Categorías de transacciones y frecuencia")
    
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Detección de anomalías para prevención de fraudes",
        "Segmentación automática por riesgo crediticio",
        "Clustering de comportamientos de gasto similares",
        "Identificación de patrones de abandono de clientes",
        "Perfilado no supervisado para productos financieros"
    ])
    
    show_subsection("Técnicas aplicables", emoji="🔬")
    show_list([
        "Isolation Forest para detección de transacciones fraudulentas",
        "K-Means para segmentación de riesgo crediticio",
        "DBSCAN para identificar comportamientos atípicos",
        "PCA para análisis de factores de riesgo principales"
    ])
    
    print()
    print("✅ Análisis teórico del Credit Card Customer Dataset completado")


def analyze_sleep_health_dataset() -> None:
    """Análisis teórico del Sleep Health and Lifestyle Dataset basado en fuente web."""
    
    show_subsection("Análisis basado en fuente web", emoji="🌐")
    print_kv("URL de origen", "https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
    print_kv("Estado del dataset", "No disponible localmente - Análisis teórico")
    
    show_subsection("Características esperadas del dataset", emoji="📊")
    print_kv("Tamaño estimado", "~400 registros de individuos")
    print_kv("Variables esperadas", "13 columnas (sueño, físicas, estilo de vida)")
    print_kv("Dominio", "Medicina del sueño y bienestar personal")
    
    show_subsection("Variables de sueño y salud esperadas", emoji="😴")
    sleep_vars = [
        "Sleep_Duration (horas de sueño por noche)",
        "Quality_of_Sleep (calidad percibida 1-10)",
        "Physical_Activity_Level (minutos ejercicio/día)",
        "Stress_Level (nivel de estrés 1-10)",
        "BMI_Category (categoría índice masa corporal)",
        "Heart_Rate (frecuencia cardíaca en bpm)",
        "Daily_Steps (pasos diarios promedio)",
        "Sleep_Disorder (trastorno del sueño diagnosticado)"
    ]
    for i, var in enumerate(sleep_vars, 1):
        print(f"  {i}. {var}")
    
    show_subsection("Perfiles de sueño esperados", emoji="🛌")
    print_kv("Cronotipos", "Mañaneros, nocturnos, intermedios")
    print_kv("Calidad del sueño", "Buena, regular, mala")
    print_kv("Trastornos", "Insomnio, apnea del sueño, sin trastornos")
    
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Clustering de cronotipos naturales sin etiquetas previas",
        "Detección de patrones de insomnio no diagnosticados",
        "Identificación de factores de riesgo combinados",
        "Segmentación por perfiles de bienestar integral",
        "Análisis de correlaciones ocultas sueño-salud"
    ])
    
    show_subsection("Técnicas recomendadas", emoji="🔬")
    show_list([
        "K-Means para tipología de dormidores",
        "Gaussian Mixture Models para cronotipos probabilísticos",
        "DBSCAN para detección de trastornos atípicos",
        "PCA para factores principales del bienestar"
    ])
    
    print()
    print("✅ Análisis teórico del Sleep Health and Lifestyle Dataset completado")


def analyze_wholesale_customers_dataset() -> None:
    """Análisis teórico del Wholesale Customers Dataset basado en fuente web."""
    
    show_subsection("Análisis basado en fuente web", emoji="🌐")
    print_kv("URL de origen", "https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set")
    print_kv("Estado del dataset", "No disponible localmente - Análisis teórico")
    
    show_subsection("Características esperadas del dataset", emoji="📊")
    print_kv("Tamaño", "440 clientes mayoristas")
    print_kv("Origen", "UCI Machine Learning Repository (dataset clásico)")
    print_kv("Sector", "Distribución B2B de productos de consumo masivo")
    print_kv("Variables", "8 columnas (Channel, Region + 6 categorías de productos)")
    
    show_subsection("Categorías de productos esperadas", emoji="📦")
    product_categories = [
        "Fresh (productos frescos: frutas, verduras, carnes)",
        "Milk (productos lácteos y derivados)",
        "Grocery (comestibles secos y enlatados)",
        "Frozen (productos congelados)",
        "Detergents_Paper (productos de limpieza y papel)",
        "Delicatessen (productos gourmet y especialidades)"
    ]
    for i, cat in enumerate(product_categories, 1):
        print(f"  {i}. {cat}")
    
    show_subsection("Segmentación comercial esperada", emoji="🏢")
    print_kv("Canales de venta", "Horeca (hoteles/restaurantes) vs Retail (minoristas)")
    print_kv("Regiones", "Diferentes áreas geográficas de distribución")
    print_kv("Patrones de compra", "Estacionales vs constantes por categoría")
    
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Segmentación clásica de clientes B2B por volumen de compra",
        "Identificación de patrones estacionales sin supervisión",
        "Clustering para estrategias de cross-selling automático",
        "Detección de clientes atípicos (outliers de compra)",
        "Análisis de canasta de mercado mayorista"
    ])
    
    show_subsection("Técnicas aplicables", emoji="🔬")
    show_list([
        "K-Means para segmentación clásica (caso de estudio típico)",
        "Hierarchical clustering para taxonomía de compradores",
        "PCA para reducción dimensional de categorías de productos",
        "DBSCAN para identificar nichos de mercado específicos"
    ])
    
    print()
    print("✅ Análisis teórico del Wholesale Customers Dataset completado")


# ======================================
# NUEVA FUNCIÓN: ANALYZE PLANTVILLAGE DATASET
# ======================================

def analyze_plantvillage_dataset(dataset_path: str) -> None:
    """Análisis completo del PlantVillage Dataset con EDA visual avanzado."""
    import os
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image, ImageStat
    from matplotlib.gridspec import GridSpec
    from collections import Counter
    
    if not os.path.exists(dataset_path):
        print("⚠️ Dataset no encontrado en la ruta especificada")
        return

    # ===== Carga y descripción ===== #
    show_subsection("Carga y descripción del conjunto de datos", emoji="✅")
    
    # Listar carpetas (clases)
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    num_classes = len(classes)
    
    print_kv("Ubicación", dataset_path)
    print_kv("Clases detectadas", num_classes)
    
    # Contar imágenes por clase y calcular estadísticas
    class_counts = {}
    total_size_bytes = 0
    
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        class_counts[cls] = len(images)
        
        # Calcular tamaño de carpeta
        for img_file in images[:5]:  # solo algunos archivos para optimizar
            try:
                total_size_bytes += os.path.getsize(os.path.join(folder, img_file))
            except:
                continue

    # Convertir a DataFrame para análisis
    df_counts = pd.DataFrame(list(class_counts.items()), columns=["Clase", "Cantidad"])
    total_images = df_counts["Cantidad"].sum()
    
    print_kv("Total de imágenes", f"{total_images:,}")
    print_kv("Promedio por clase", f"{df_counts['Cantidad'].mean():.1f}")
    print_kv("Desviación estándar", f"{df_counts['Cantidad'].std():.1f}")
    print_kv("Tamaño estimado", f"{(total_size_bytes * total_images / (5 * len(classes))) / (1024**2):.1f} MB")

    # ===== Análisis de distribución ===== #
    show_subsection("Análisis de distribución por clases", emoji="📊")
    
    # Separar por tipo de planta y condición
    plant_types = {}
    conditions = Counter()
    
    for cls in classes:
        if '___' in cls:
            plant, condition = cls.split('___', 1)
            if plant not in plant_types:
                plant_types[plant] = []
            plant_types[plant].append(condition)
            conditions[condition] += class_counts[cls]
        else:
            # Para clases sin separador claro
            parts = cls.split('_')
            if len(parts) > 1:
                condition = parts[-1] if 'healthy' in parts[-1].lower() or 'disease' in parts[-1].lower() else 'other'
                conditions[condition] += class_counts[cls]
    
    print_kv("Tipos de plantas", len(plant_types))
    print_kv("Condiciones detectadas", len(conditions))
    print_kv("Distribución de condiciones", dict(conditions.most_common(5)))

    # ===== EDA Visual Avanzado ===== #
    show_subsection("EDA Visual", emoji="🎨")
    
    # Crear figura con grid personalizado
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1️⃣ Distribución por clases (top 10)
    ax1 = fig.add_subplot(gs[0, :2])
    top_classes = df_counts.nlargest(10, 'Cantidad')
    bars = ax1.barh(top_classes['Clase'], top_classes['Cantidad'], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_classes))))
    ax1.set_title('Top 10 Clases por Cantidad de Imágenes', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cantidad de Imágenes')
    
    # Agregar valores en las barras
    for i, (bar, value) in enumerate(zip(bars, top_classes['Cantidad'])):
        ax1.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                str(value), va='center', ha='left', fontweight='bold')
    
    # 2️⃣ Histograma de distribución
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.hist(df_counts['Cantidad'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(df_counts['Cantidad'].mean(), color='red', linestyle='--', 
               label=f'Media: {df_counts["Cantidad"].mean():.0f}')
    ax2.axvline(df_counts['Cantidad'].median(), color='orange', linestyle='--',
               label=f'Mediana: {df_counts["Cantidad"].median():.0f}')
    ax2.set_title('Distribución de Imágenes por Clase', fontweight='bold')
    ax2.set_xlabel('Cantidad de Imágenes')
    ax2.set_ylabel('Frecuencia')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3️⃣ Mosaico de imágenes representativas
    ax3 = fig.add_subplot(gs[1:3, :])
    ax3.axis('off')
    ax3.set_title('Mosaico de Muestras Representativas por Clase', fontsize=14, fontweight='bold', pad=20)
    
    # Seleccionar 12 clases aleatorias para el mosaico
    sample_classes = random.sample(classes, min(12, len(classes)))
    
    for i, cls in enumerate(sample_classes):
        try:
            folder = os.path.join(dataset_path, cls)
            images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                sample_image = random.choice(images)
                img_path = os.path.join(folder, sample_image)
                img = Image.open(img_path)
                
                # Crear subplot para cada imagen
                row = i // 4 + 1
                col = i % 4
                ax_img = plt.subplot(4, 4, row * 4 + col + 1)
                ax_img.imshow(img)
                ax_img.axis('off')
                
                # Título con información de la clase
                title = cls.replace('___', '\n').replace('_', ' ')
                if len(title) > 25:
                    title = title[:22] + "..."
                ax_img.set_title(title, fontsize=8, fontweight='bold')
        except Exception as e:
            continue
    
    # 4️⃣ Análisis de características de imagen
    ax4 = fig.add_subplot(gs[3, :2])
    
    # Calcular estadísticas de color y tamaño
    brightness_data = []
    size_data = []
    aspect_ratios = []
    
    sample_for_analysis = random.sample(classes, min(5, len(classes)))
    
    for cls in sample_for_analysis:
        folder = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        for img_file in random.sample(images, min(3, len(images))):
            try:
                img_path = os.path.join(folder, img_file)
                with Image.open(img_path) as img:
                    # Brillo promedio
                    stat = ImageStat.Stat(img)
                    brightness = sum(stat.mean) / len(stat.mean)
                    brightness_data.append(brightness)
                    
                    # Dimensiones
                    width, height = img.size
                    size_data.append(width * height)
                    aspect_ratios.append(width / height)
            except:
                continue
    
    # Gráfico de dispersión: tamaño vs brillo
    scatter = ax4.scatter(size_data, brightness_data, c=aspect_ratios, 
                         cmap='viridis', alpha=0.7, s=60)
    ax4.set_xlabel('Tamaño (píxeles)')
    ax4.set_ylabel('Brillo Promedio')
    ax4.set_title('Características de Imagen: Tamaño vs Brillo', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Colorbar para aspect ratio
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Aspect Ratio')
    
    # 5️⃣ Distribución de condiciones (si aplicable)
    ax5 = fig.add_subplot(gs[3, 2:])
    
    if len(conditions) > 1:
        condition_names = list(conditions.keys())
        condition_counts = list(conditions.values())
        
        wedges, texts, autotexts = ax5.pie(condition_counts, labels=condition_names, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=plt.cm.Set3(np.linspace(0, 1, len(condition_names))))
        ax5.set_title('Distribución por Condición de Salud', fontweight='bold')
        
        # Mejorar legibilidad del texto
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax5.text(0.5, 0.5, 'Análisis de condiciones\nno disponible', 
                ha='center', va='center', transform=ax5.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax5.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ===== Estadísticas avanzadas ===== #
    show_subsection("Estadísticas técnicas", emoji="📐")
    
    if size_data and brightness_data:
        print_kv("Resolución promedio", f"{np.mean([np.sqrt(s) for s in size_data]):.0f} px (equiv.)")
        print_kv("Brillo promedio", f"{np.mean(brightness_data):.1f}")
        print_kv("Aspect ratio promedio", f"{np.mean(aspect_ratios):.2f}")
        print_kv("Variabilidad de tamaño", f"CV = {np.std(size_data)/np.mean(size_data)*100:.1f}%")
    
    # ===== Potencial para ML No Supervisado ===== #
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "Clustering visual por características de color y textura",
        "Detección de anomalías en hojas con patrones atípicos",
        "Reducción dimensional con autoencoders para embeddings de imágenes",
        "Segmentación no supervisada por severidad de enfermedad",
        "Análisis de componentes principales en características visuales"
    ])
    
    show_subsection("Técnicas recomendadas", emoji="🔬")
    show_list([
        "K-Means sobre características CNN pre-entrenadas (ResNet, VGG)",
        "DBSCAN para detección de imágenes outliers o mal etiquetadas",
        "t-SNE/UMAP para visualización de clusters en espacio de características",
        "Autoencoders variacionales para generación y clustering latente"
    ])
    
    print()
    print("✅ Análisis del PlantVillage Dataset completado exitosamente")


# ======================================
# NUEVA FUNCIÓN: ANALYZE MILK QUALITY DATASET
# ======================================

def analyze_milk_quality_dataset(file_path: str) -> None:
    """Análisis completo del Milk Quality Dataset con EDA visual avanzado."""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    
    try:
        # ===== Carga y descripción ===== #
        df = pd.read_csv(file_path)
        
        show_subsection("Carga y descripción del conjunto de datos", emoji="✅")
        print_kv("Ubicación", file_path)
        print_kv("Tamaño", f"{df.shape[0]:,} filas × {df.shape[1]} columnas")
        print_kv("Uso de memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        show_subsection("Columnas disponibles", emoji="📋")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique() if df[col].dtype in ['object', 'int64'] else 'continua'
            print(f"  {i:2d}. {col:25s} ({dtype:8s}) - {unique_vals} valores únicos")

        # ===== Análisis estadístico ===== #
        show_subsection("Análisis estadístico", emoji="📈")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print_kv("Variables numéricas", len(numeric_cols))
        print_kv("Variables categóricas", len(categorical_cols))
        
        # Estadísticas básicas para variables numéricas
        if numeric_cols:
            for col in numeric_cols[:6]:  # Mostrar primeras 6
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val, max_val = df[col].min(), df[col].max()
                print(f"  * {col}: μ={mean_val:.2f} (±{std_val:.2f}), rango=[{min_val:.2f}, {max_val:.2f}]")

        # Verificar valores nulos y completitud
        null_counts = df.isnull().sum()
        completeness = (1 - null_counts.sum() / (df.shape[0] * df.shape[1])) * 100
        print_kv("Completitud", f"{completeness:.1f}%")
        
        if null_counts.sum() > 0:
            print_kv("Columnas con nulos", {col: count for col, count in null_counts.items() if count > 0})

        # ===== EDA Visual Avanzado ===== #
        show_subsection("EDA Visual", emoji="📊")
        
        # Crear figura con grid complejo
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1️⃣ Distribuciones de variables principales
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Identificar variables principales (primeras numéricas)
        main_vars = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        
        for i, var in enumerate(main_vars):
            ax1.hist(df[var], alpha=0.6, label=var, bins=20)
        
        ax1.set_title('Distribuciones de Variables Fisicoquímicas Principales', fontweight='bold')
        ax1.set_xlabel('Valores')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2️⃣ Matriz de correlación
        ax2 = fig.add_subplot(gs[0, 2:])
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                       center=0, ax=ax2, fmt='.2f', linewidths=0.5)
            ax2.set_title('Matriz de Correlación entre Variables', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Matriz de correlación\nno disponible', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3️⃣ Boxplots para detectar outliers
        ax3 = fig.add_subplot(gs[1, :])
        
        if main_vars:
            # Normalizar datos para mejor visualización
            df_norm = df[main_vars].copy()
            for col in main_vars:
                df_norm[col] = (df[col] - df[col].mean()) / df[col].std()
            
            bp = ax3.boxplot([df_norm[col].dropna() for col in main_vars], 
                            labels=main_vars, patch_artist=True)
            
            # Colorear boxplots
            colors = plt.cm.Set3(np.linspace(0, 1, len(main_vars)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_title('Detección de Outliers (Valores Normalizados)', fontweight='bold')
            ax3.set_ylabel('Valores Z-Score')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4️⃣ Análisis de calidad (si existe variable de calidad)
        ax4 = fig.add_subplot(gs[2, :2])
        
        # Buscar variable de calidad/grade
        quality_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['grade', 'quality', 'class', 'type']):
                quality_col = col
                break
        
        if quality_col and quality_col in categorical_cols:
            quality_counts = df[quality_col].value_counts()
            wedges, texts, autotexts = ax4.pie(quality_counts.values, 
                                              labels=quality_counts.index,
                                              autopct='%1.1f%%', startangle=90,
                                              colors=plt.cm.Pastel1(np.linspace(0, 1, len(quality_counts))))
            ax4.set_title(f'Distribución de {quality_col}', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, 'Análisis de calidad\nno disponible\n(variable no identificada)', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.axis('off')
        
        # 5️⃣ Scatter plot multivariado
        ax5 = fig.add_subplot(gs[2, 2:])
        
        if len(main_vars) >= 2:
            x_var, y_var = main_vars[0], main_vars[1]
            
            # Color por tercera variable si está disponible
            if len(main_vars) >= 3:
                scatter = ax5.scatter(df[x_var], df[y_var], c=df[main_vars[2]], 
                                    cmap='viridis', alpha=0.6, s=50)
                plt.colorbar(scatter, ax=ax5, label=main_vars[2])
            else:
                ax5.scatter(df[x_var], df[y_var], alpha=0.6, s=50, color='skyblue')
            
            ax5.set_xlabel(x_var)
            ax5.set_ylabel(y_var)
            ax5.set_title(f'Relación {x_var} vs {y_var}', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Agregar línea de tendencia
            if df[x_var].notna().sum() > 1 and df[y_var].notna().sum() > 1:
                z = np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1)
                p = np.poly1d(z)
                ax5.plot(df[x_var].dropna().sort_values(), p(df[x_var].dropna().sort_values()), 
                        "r--", alpha=0.8, label=f'Tendencia (R²≈{np.corrcoef(df[x_var].dropna(), df[y_var].dropna())[0,1]**2:.3f})')
                ax5.legend()
        
        # 6️⃣ Análisis de normalidad
        ax6 = fig.add_subplot(gs[3, :])
        
        if main_vars:
            # Test de normalidad para variables principales
            normality_results = []
            
            for var in main_vars[:4]:  # máximo 4 variables
                data = df[var].dropna()
                if len(data) > 3:
                    statistic, p_value = stats.shapiro(data[:5000])  # límite para performance
                    is_normal = p_value > 0.05
                    normality_results.append({
                        'Variable': var,
                        'Statistic': statistic,
                        'P-value': p_value,
                        'Normal': 'Sí' if is_normal else 'No'
                    })
            
            # Crear tabla de normalidad
            if normality_results:
                norm_df = pd.DataFrame(normality_results)
                ax6.axis('tight')
                ax6.axis('off')
                table = ax6.table(cellText=norm_df.values,
                                colLabels=norm_df.columns,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                
                # Colorear según normalidad
                for i in range(len(norm_df)):
                    color = 'lightgreen' if norm_df.iloc[i]['Normal'] == 'Sí' else 'lightcoral'
                    table[(i+1, 3)].set_facecolor(color)  # Columna 'Normal'
                
                ax6.set_title('Test de Normalidad (Shapiro-Wilk)', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        # ===== Estadísticas avanzadas ===== #
        show_subsection("Estadísticas técnicas", emoji="📐")
        
        if numeric_cols:
            print_kv("Coeficiente de variación promedio", f"{np.mean([df[col].std()/df[col].mean() for col in numeric_cols if df[col].mean() != 0]):.3f}")
            
            # Detectar outliers usando IQR
            total_outliers = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col].count()
                total_outliers += outliers
            
            print_kv("Outliers detectados (IQR)", f"{total_outliers} valores ({total_outliers/len(df)*100:.1f}%)")
            
            # Análisis de calidad si está disponible
            if quality_col:
                print_kv("Variable de calidad identificada", quality_col)
                print_kv("Categorías de calidad", list(df[quality_col].unique()))

    except FileNotFoundError:
        print("⚠️ Archivo no encontrado en la ruta especificada")
        
        show_subsection("Especificaciones del conjunto de datos (referencia)", emoji="📋")
        show_list([
            "Variables típicas: pH, Temperatura, Sabor, Olor, Grasa, Turbidez",
            "Rango pH: 3.0-9.5 (acidez de la leche)",
            "Temperatura: 34-90°C (procesamiento)",
            "Contenido graso: 0-7% (diferentes tipos de leche)",
        ])
    
    except Exception as e:
        print(f"⚠️ Error al procesar el dataset: {str(e)}")
    
    # ===== Potencial para ML No Supervisado ===== #
    show_subsection("Potencial para ML no supervisado", emoji="🎯")
    show_list([
        "K-Means para categorización automática de calidad (Alta/Media/Baja)",
        "Isolation Forest para detección de contaminación bacteriana",
        "DBSCAN para identificar lotes con características excepcionales",
        "PCA para identificar los 2-3 factores más críticos de calidad",
        "Gaussian Mixture Models para modelar distribución natural de calidad"
    ])

    show_subsection("Técnicas recomendadas", emoji="🔬")
    show_list([
        "Normalización Z-score antes de clustering por diferencias de escala",
        "PCA para reducir dimensionalidad de parámetros fisicoquímicos",
        "K-Means con elbow method para determinar clusters óptimos",
        "Análisis de componentes independientes (ICA) para factores latentes"
    ])
    
    show_subsection("Aplicación en entornos productivos", emoji="🏭")
    show_list([
        "Sistema de clasificación automática en línea de producción",
        "Alertas tempranas de desviaciones en calidad",
        "Optimización de parámetros de procesamiento",
        "Trazabilidad y control de calidad predictivo"
    ], bullet="•")
    
    print()
    print("✅ Análisis del Milk Quality Dataset completado exitosamente")


def evaluate_datasets_comparative():
    """
    Función para evaluación comparativa de los 10 datasets del proyecto.
    
    Evalúa cada dataset según criterios específicos para el contexto del sur de Chile:
    - Diversidad de Datos: Variedad en tipos de datos y estructura
    - Potencial Clustering: Capacidad para generar agrupamientos significativos  
    - Relevancia Impacto: Aplicabilidad en el contexto rural del sur de Chile
    - Disponibilidad: Acceso real a los datos para el proyecto
    - Complementariedad: Sinergia con otros datasets del ecosistema rural
    
    Los puntajes se asignan en escala 1-10 considerando el objetivo principal:
    desarrollo de inteligencia para economía circular en bosques comestibles,
    monitoreo ganadero y control de calidad láctea en el sur de Chile.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Criterios de evaluación contextualizados al sur de Chile
    # Puntajes basados en discusiones del equipo considerando:
    # - Aplicabilidad en contexto rural sur de Chile
    # - Integración con sistemas de economía circular  
    # - Potencial para inteligencia distribuida con sensores IoT
    # - Relevancia para productores lecheros y forestales
    evaluacion_datasets = {
        'No.': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Dataset': [
            'Dairy Sales',           # 1
            'Animal Sounds',         # 2  
            'Customer Personality',  # 3
            'News Category',         # 4
            'Coffee Health',         # 5
            'Credit Card',           # 6
            'Sleep Health',          # 7
            'Wholesale Customers',   # 8
            'PlantVillage',         # 9
            'Milk Quality'           # 10
        ],
        # Diversidad_Datos: Variedad en estructura y tipos de datos (1-10)
        'Diversidad_Datos': [6, 9, 8, 8, 7, 7, 6, 6, 9, 8],
        
        # Potencial_Clustering: Capacidad de generar clusters significativos (1-10)
        'Potencial_Clustering': [7, 9, 8, 6, 7, 8, 7, 8, 8, 8],
        
        # Relevancia_Impacto: Aplicabilidad directa en el sur de Chile (1-10)
        # Priorizamos: ganadería lechera, agricultura, bienestar animal
        'Relevancia_Impacto': [6, 9, 7, 4, 6, 5, 6, 6, 9, 9],
        
        # Disponibilidad: Acceso garantizado a datos (1-10)
        'Disponibilidad': [5, 9, 5, 5, 5, 5, 5, 5, 9, 10],
        
        # Complementariedad: Sinergia con ecosistema rural inteligente (1-10)
        'Complementariedad': [6, 9, 6, 4, 5, 5, 6, 6, 9, 9]
    }

    df_eval = pd.DataFrame(evaluacion_datasets)

    # Calcular puntuación total ponderada
    # Priorizamos relevancia e impacto para el contexto del sur de Chile
    df_eval['Puntuacion_Total'] = (
        df_eval['Diversidad_Datos'] * 0.15 +
        df_eval['Potencial_Clustering'] * 0.20 +
        df_eval['Relevancia_Impacto'] * 0.35 +  # Mayor peso por contexto rural
        df_eval['Disponibilidad'] * 0.15 +
        df_eval['Complementariedad'] * 0.15
    )

    # Ordenar por puntuación total
    df_eval = df_eval.sort_values('Puntuacion_Total', ascending=False)

    show_section("Evaluación Comparativa de Datasets", emoji="📊")
    
    # Mostrar contexto de evaluación
    print("🎯 Criterios de Evaluación para Inteligencia Rural del Sur de Chile:")
    print("   • Diversidad de Datos (15%): Variedad en estructura y tipos")
    print("   • Potencial Clustering (20%): Capacidad de agrupamientos significativos") 
    print("   • Relevancia e Impacto (35%): Aplicabilidad directa en contexto rural")
    print("   • Disponibilidad (15%): Acceso garantizado a los datos")
    print("   • Complementariedad (15%): Sinergia con ecosistema inteligente")
    print()
    
    # Mostrar tabla de evaluación
    display(df_eval.round(2))

    # Crear visualizaciones
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Ranking por puntuación total
    colors = plt.cm.viridis(df_eval['Puntuacion_Total']/df_eval['Puntuacion_Total'].max())
    ax1.barh(df_eval['Dataset'], df_eval['Puntuacion_Total'], color=colors)
    ax1.set_xlabel('Puntuación Total')
    ax1.set_title('🏆 Ranking de Datasets - Contexto Sur de Chile')
    ax1.grid(True, alpha=0.3)
    
    # Anotar puntajes
    for i, (dataset, score) in enumerate(zip(df_eval['Dataset'], df_eval['Puntuacion_Total'])):
        ax1.text(score + 0.1, i, f'{score:.1f}', va='center', ha='left', fontweight='bold')

    # 2. Comparación por criterios (Top 5)
    top_5 = df_eval.head(5)
    criterios = ['Diversidad_Datos', 'Potencial_Clustering', 'Relevancia_Impacto', 
                'Disponibilidad', 'Complementariedad']
    
    x = range(len(top_5))
    width = 0.15
    
    for i, criterio in enumerate(criterios):
        ax2.bar([pos + i*width for pos in x], top_5[criterio], 
                width, label=criterio.replace('_', ' '), alpha=0.8)
    
    ax2.set_xlabel('Datasets (Top 5)')
    ax2.set_ylabel('Puntaje (1-10)')
    ax2.set_title('📋 Comparación por Criterios - Top 5 Datasets')
    ax2.set_xticks([pos + width*2 for pos in x])
    ax2.set_xticklabels(top_5['Dataset'], rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. Distribución por dominio (contextualizada)
    dominios_contexto = {
        'Agropecuario': ['Animal Sounds', 'PlantVillage', 'Milk Quality'],  # Relevantes
        'Comercial': ['Dairy Sales', 'Customer Personality', 'Wholesale Customers'],
        'Salud/Bienestar': ['Coffee Health', 'Sleep Health'], 
        'Otros': ['News Category', 'Credit Card']
    }
    
    conteos_contexto = [len(datasets) for datasets in dominios_contexto.values()]
    colores_contexto = ['#2E8B57', '#4682B4', '#DAA520', '#CD5C5C']  # Verde para agropecuario
    
    wedges, texts, autotexts = ax3.pie(conteos_contexto, labels=list(dominios_contexto.keys()), 
                                      autopct='%1.1f%%', startangle=90, colors=colores_contexto)
    ax3.set_title('🌱 Distribución por Relevancia en Contexto Rural')
    
    # Destacar sector agropecuario
    wedges[0].set_linewidth(3)
    wedges[0].set_edgecolor('darkgreen')

    # 4. Matriz de correlación entre criterios
    criterios_df = df_eval[criterios].corr()
    im = ax4.imshow(criterios_df, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(criterios)))
    ax4.set_yticks(range(len(criterios)))
    ax4.set_xticklabels([c.replace('_', '\n') for c in criterios], rotation=45)
    ax4.set_yticklabels([c.replace('_', '\n') for c in criterios])
    ax4.set_title('🔗 Correlación entre Criterios de Evaluación')
    
    # Añadir valores de correlación
    for i in range(len(criterios)):
        for j in range(len(criterios)):
            ax4.text(j, i, f'{criterios_df.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im, ax=ax4, shrink=0.6)

    plt.tight_layout()
    plt.show()

    # Mostrar resultados contextualizados
    show_subsection("🎯 Top 3 Datasets para Inteligencia Rural", emoji="🏆")
    top_3 = df_eval.head(3)
    
    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
        relevancia = "🌟 ALTA RELEVANCIA" if row['Relevancia_Impacto'] >= 8 else "⚡ RELEVANCIA MEDIA" if row['Relevancia_Impacto'] >= 6 else "📋 RELEVANCIA BÁSICA"
        print(f"{i}. {row['Dataset']} - Puntuación: {row['Puntuacion_Total']:.2f} - {relevancia}")

    show_subsection("💡 Justificación Estratégica para el Sur de Chile", emoji="🌱")
    show_list([
        "🎵 Animal Sounds: Monitoreo bienestar animal y detección automática de alertas",
        "🥛 Milk Quality: Control de calidad automatizado en líneas de producción láctea", 
        "🌿 PlantVillage: Diagnóstico temprano de enfermedades en cultivos y bosques comestibles",
        "📡 Complementariedad IoT: Integración natural con sensores distribuidos",
        "🔄 Economía Circular: Optimización de recursos en sistemas agropecuarios sustentables"
    ])
    
    show_subsection("🎯 Alineación con Objetivos del Proyecto", emoji="📊") 
    print("• Diversidad técnica máxima: Audio (señales) + Numérico (fisicoquímico) + Visual (RGB)")
    print("• Aplicabilidad directa en ganadería lechera del sur de Chile")
    print("• Potencial para integración con robótica e IoT distribuido")
    print("• Base para inteligencia de economía circular en entornos rurales")
    
    return df_eval


def generate_datasets_summary_table():
    """
    Genera y muestra la tabla resumen de los 10 datasets con información completa
    incluyendo descripción, dominio, tipo de datos y aplicaciones en ML no supervisado.
    """
    datasets_info = {
        "No.": list(range(1, 11)),
        "Dataset": [
            "Dairy Goods Sales Dataset", "Animal Sounds Dataset",
            "Customer Personality Analysis", "News Category Dataset",
            "Global Coffee Health Dataset", "Credit Card Customer",
            "Sleep Health and Lifestyle", "Wholesale Customers Dataset",
            "PlantVillage Dataset", "Milk Quality Dataset"
        ],
        "Dominio": [
            "Comercio/Ventas", "Audio/Sonidos", "Comportamiento del Consumidor",
            "Texto/NLP", "Salud Global", "Servicios Financieros",
            "Salud y Bienestar", "Comercio B2B", "Visión por Computador",
            "Calidad Alimentaria"
        ],
        "Tipo_de_Datos": [
            "Transaccional", "Audio (WAV)", "Demográfico/Comportamental",
            "Texto/JSON", "Numérico/Geográfico", "Financiero/Numérico",
            "Médico/Estilo de vida", "Ventas por categoría",
            "Imágenes RGB", "Fisicoquímico/Numérico"
        ],
        "Formato_Archivo": [
            "CSV", "WAV/Audio", "CSV", "JSON", "CSV",
            "CSV", "CSV", "CSV", "JPG/PNG", "CSV"
        ],
        "URL_Origen": [
            "https://www.kaggle.com/datasets/suraj520/dairy-goods-sales-dataset",
            "https://github.com/YashNita/Animal-Sound-Dataset",
            "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis",
            "https://www.kaggle.com/datasets/rmisra/news-category-dataset",
            "https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset",
            "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers",
            "https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset",
            "https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set",
            "https://www.kaggle.com/datasets/plantvillage-dataset",
            "https://www.kaggle.com/datasets/cpluzshrijayan/milkquality"
        ],
        "Aplicaciones_ML_No_Supervisado": [
            "Segmentación clientes, Patrones compra",
            "Clustering sonidos, Clasificación no supervisada",
            "Segmentación personalidad, Perfiles cliente",
            "Topic modeling, Clustering temas",
            "Clustering geográfico, Patrones consumo",
            "Detección fraudes, Clustering crediticio",
            "Patrones sueño, Clustering estilos vida",
            "Segmentación B2B, Patrones compra",
            "Clustering visual, Detección anomalías",
            "Control calidad, Detección anomalías"
        ]
    }

    df_summary = pd.DataFrame(datasets_info)

    # 💅 Estilo visual con pandas Styler
    styled_df = (
        df_summary.style
        .set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#4C72B0"), ("color", "white"), ("font-weight", "bold")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
            {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "white")]}
        ])
        .set_properties(**{
            "text-align": "left",
            "border": "1px solid #ddd",
            "color": "#222222"   # 🔹 color de letra más oscuro dentro de las celdas
        })
        .hide(axis="index")
    )

    show_section("Tabla resumen de los 10 datasets", emoji="📋")
    display(styled_df)

    # 🧾 Resumen general
    blank()
    print_kv("Total de datasets analizados", len(df_summary))
    print_kv("Dominios cubiertos", df_summary["Dominio"].nunique())
    print_kv("Tipos de formato", ", ".join(df_summary["Formato_Archivo"].unique()))
    
    return df_summary
