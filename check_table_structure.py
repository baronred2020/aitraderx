#!/usr/bin/env python3
"""
Script para verificar la estructura real de la tabla user_predictions
"""

import os
import sys

# Agregar el directorio backend/src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from config.database_config import db_config

def check_table_structure():
    """
    Verificar la estructura real de la tabla user_predictions
    """
    print("🔍 VERIFICACIÓN DE ESTRUCTURA DE TABLA")
    print("=" * 80)
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar si la tabla existe
            cursor.execute("SHOW TABLES LIKE 'user_predictions'")
            if not cursor.fetchone():
                print("❌ La tabla user_predictions no existe")
                return
            
            print("✅ Tabla user_predictions existe")
            
            # Obtener estructura completa
            cursor.execute("DESCRIBE user_predictions")
            columns = cursor.fetchall()
            
            print("\n📋 Estructura completa de la tabla:")
            print("-" * 80)
            for col in columns:
                print(f"   - {col[0]}: {col[1]} {'NULL' if col[2] == 'YES' else 'NOT NULL'} {'DEFAULT ' + str(col[4]) if col[4] else ''}")
            
            # Verificar si existen las columnas que necesitamos
            column_names = [col[0] for col in columns]
            
            required_columns = [
                'id', 'user_id', 'pair', 'direction', 'current_price', 
                'confidence', 'precision', 'win_rate', 'timeframe', 
                'reasoning', 'brain_type', 'created_at', 'expires_at',
                'is_completed', 'actual_price_at_expiry', 'prediction_success', 
                'success_percentage'
            ]
            
            print("\n🔍 Verificando columnas requeridas:")
            missing_columns = []
            for col in required_columns:
                if col in column_names:
                    print(f"   ✅ {col}")
                else:
                    print(f"   ❌ {col} - FALTANTE")
                    missing_columns.append(col)
            
            if missing_columns:
                print(f"\n⚠️  Columnas faltantes: {missing_columns}")
                
                # Crear script SQL para agregar columnas faltantes
                print("\n📝 Script SQL para agregar columnas faltantes:")
                for col in missing_columns:
                    if col == 'target_price':
                        print(f"ALTER TABLE user_predictions ADD COLUMN {col} DECIMAL(10,5) NULL;")
                    elif col in ['precision', 'win_rate']:
                        print(f"ALTER TABLE user_predictions ADD COLUMN `{col}` DECIMAL(5,2) DEFAULT 0.0;")
                    else:
                        print(f"ALTER TABLE user_predictions ADD COLUMN {col} VARCHAR(50) NULL;")
            else:
                print("\n✅ Todas las columnas requeridas están presentes")
            
            cursor.close()
            
    except Exception as e:
        print(f"❌ Error verificando estructura: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_table_structure()