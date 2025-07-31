#!/usr/bin/env python3
"""
Script de prueba para verificar la detección de Colab y acceso a archivos
"""

import os
import sys

def is_colab_environment():
    """Detectar si estamos en Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_kaggle_environment():
    """Detectar si estamos en Kaggle"""
    return os.path.exists("/kaggle/input")

def test_file_access():
    """Probar acceso a archivos en diferentes entornos"""
    print("🔍 PROBANDO DETECCIÓN DE ENTORNOS Y ACCESO A ARCHIVOS")
    print("=" * 60)
    
    # Detectar entorno
    if is_colab_environment():
        print("☁️ Detectado: Google Colab")
        content_path = "/content"
        if os.path.exists(content_path):
            print(f"✅ Directorio {content_path} existe")
            files = os.listdir(content_path)
            print(f"📁 Archivos en /content: {files}")
            
            csv_files = [f for f in files if f.lower().endswith('.csv')]
            if csv_files:
                print(f"📄 Archivos CSV encontrados: {csv_files}")
                for csv_file in csv_files:
                    file_path = os.path.join(content_path, csv_file)
                    file_size = os.path.getsize(file_path)
                    print(f"   - {csv_file}: {file_size} bytes")
            else:
                print("❌ No se encontraron archivos CSV en /content")
        else:
            print(f"❌ Directorio {content_path} no existe")
    
    elif is_kaggle_environment():
        print("🌐 Detectado: Kaggle")
        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            print(f"✅ Directorio {kaggle_input} existe")
            datasets = os.listdir(kaggle_input)
            print(f"📁 Datasets en Kaggle: {datasets}")
        else:
            print(f"❌ Directorio {kaggle_input} no existe")
    
    else:
        print("🏠 Detectado: Entorno local")
    
    # Verificar directorio actual
    current_dir = os.getcwd()
    print(f"\n📂 Directorio actual: {current_dir}")
    current_files = os.listdir(current_dir)
    csv_files_current = [f for f in current_files if f.lower().endswith('.csv')]
    if csv_files_current:
        print(f"📄 Archivos CSV en directorio actual: {csv_files_current}")
    else:
        print("❌ No se encontraron archivos CSV en el directorio actual")
    
    print("\n" + "=" * 60)
    print("✅ PRUEBA COMPLETADA")

if __name__ == "__main__":
    test_file_access() 