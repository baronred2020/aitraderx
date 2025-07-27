#!/usr/bin/env python3
"""
Script para verificar las tablas existentes en trading_db
"""
import mysql.connector
from mysql.connector import Error

try:
    # Conectar a la base de datos
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='trading_db'
    )
    
    if connection.is_connected():
        cursor = connection.cursor()
        
        # Mostrar todas las tablas
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("Tablas existentes en trading_db:")
        for table in tables:
            print(f"- {table[0]}")
        
        # Verificar si existe la tabla predictions
        cursor.execute("SHOW TABLES LIKE 'predictions'")
        predictions_table = cursor.fetchall()
        
        if predictions_table:
            print("\n✅ Tabla 'predictions' existe")
            cursor.execute("DESCRIBE predictions")
            columns = cursor.fetchall()
            print("Estructura de la tabla predictions:")
            for col in columns:
                print(f"  {col[0]}: {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
        else:
            print("\n❌ Tabla 'predictions' NO existe")
        
        # Verificar si existe la tabla user_predictions
        cursor.execute("SHOW TABLES LIKE 'user_predictions'")
        user_predictions_table = cursor.fetchall()
        
        if user_predictions_table:
            print("\n✅ Tabla 'user_predictions' existe")
            cursor.execute("DESCRIBE user_predictions")
            columns = cursor.fetchall()
            print("Estructura de la tabla user_predictions:")
            for col in columns:
                print(f"  {col[0]}: {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
        else:
            print("\n❌ Tabla 'user_predictions' NO existe")
            
        cursor.close()
        connection.close()
        print("\n✅ Conexión cerrada")

except Error as e:
    print(f"Error conectando a MySQL: {e}")
except Exception as e:
    print(f"Error: {e}") 