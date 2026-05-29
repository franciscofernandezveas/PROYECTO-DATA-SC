import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proceso_etl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_fecha(fecha_str):
    """Función auxiliar para parsear fechas en múltiples formatos"""
    if pd.isna(fecha_str):
        return pd.NaT
    
    formatos = [
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y'
    ]
    
    fecha_str = str(fecha_str).strip()
    
    for formato in formatos:
        try:
            return pd.to_datetime(fecha_str, format=formato, dayfirst=True)
        except:
            continue
    
    # Intentar parseo automático
    try:
        return pd.to_datetime(fecha_str, dayfirst=True)
    except:
        return pd.NaT

def registrar_etapa(df, nombre_etapa, stats):
    """Función auxiliar para registrar estadísticas"""
    if nombre_etapa not in stats:
        stats[nombre_etapa] = len(df)
    return df

# =====================================================================
# FUNCIÓN DE LIMPIEZA PARA INFORME_VENTAS
# =====================================================================
def limpiar_informe_ventas(input_file, output_file=None, eliminar_outliers=True):
    """
    Limpia el archivo informe_ventas y devuelve el DataFrame limpio
    """
    logger.info("=== INICIANDO LIMPIEZA DE INFORME_VENTAS ===")
    stats = {}

    # 1️⃣ CARGA DE DATOS
    logger.info("1. Cargando datos...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        stats['filas_iniciales'] = len(df)
        stats['columnas_iniciales'] = len(df.columns)
        logger.info(f"   [OK] Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        logger.error(f"   [ERROR] Error cargando archivo: {e}")
        return None, {}

    df = registrar_etapa(df, "despues_carga", stats)

    # 2️⃣ NORMALIZACIÓN DE TEXTOS
    logger.info("\n2. Normalizando textos...")
    columnas_texto = df.select_dtypes(include=['object']).columns.tolist()

    columnas_excluir_numeros = [
        'Cantidad', 'Precio sin descuento', 'Descuento',
        'Precio (Bruto)', 'Precio (Neto)', 'IVA'
    ]
    columnas_texto_filtradas = [
        col for col in columnas_texto if col not in columnas_excluir_numeros
    ]

    for columna in columnas_texto_filtradas:
        if columna in df.columns:
            df[columna] = df[columna].astype(str)
            df[columna] = (
                df[columna]
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
                .str.strip()
            )
            df[columna] = df[columna].replace(
                ['none', 'null', 'nan', ''], 'no_especificado'
            )
    logger.info("   [OK] Normalización de textos completada")

    # 3️⃣ ELIMINACIÓN DE DUPLICADOS
    logger.info("\n3. Eliminando duplicados...")
    filas_antes = len(df)
    df = df.drop_duplicates()
    duplicados_eliminados = filas_antes - len(df)
    stats['duplicados_eliminados'] = duplicados_eliminados
    logger.info(f"   [OK] Duplicados eliminados: {duplicados_eliminados}")
    df = registrar_etapa(df, "despues_duplicados", stats)

    # 4️⃣ NORMALIZACIÓN DE NOMBRES DE COLUMNAS
    logger.info("\n4. Normalizando nombres de columnas...")
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('(', '')
        .str.replace(')', '')
        .str.replace('-', '_')
    )
    logger.info(f"   Columnas normalizadas: {list(df.columns)}")

    # 5️⃣ MANEJO INTELIGENTE DE FECHAS
    logger.info("\n5. Manejo inteligente de fechas...")
    if 'Fecha' in df.columns:
        try:
            df['fecha_original_valor'] = df['Fecha'].copy()
            df['Fecha'] = df['Fecha'].apply(parse_fecha)
            
            fechas_invalidas = df['Fecha'].isnull().sum()
            stats['fechas_invalidas_inicial'] = fechas_invalidas
            logger.info(f"   [INFO] Fechas nulas detectadas: {fechas_invalidas}")

            df['fecha_problema'] = df['Fecha'].isnull()

            if len(df['Fecha'].dropna()) > 0:
                fecha_min = df['Fecha'].dropna().min()
                fecha_max = df['Fecha'].dropna().max()
                logger.info(f"   [INFO] Rango de fechas: {fecha_min} a {fecha_max}")
            
            logger.info("   [OK] Manejo de fechas completado")
        except Exception as e:
            logger.error(f"   [ERROR] Error manejando fechas: {e}")
            df['fecha_problema'] = False

    # 6️⃣ VALIDACIÓN DE CAMPOS NUMÉRICOS
    logger.info("\n6. Validando campos numéricos...")
    campos_numericos = [
        'Cantidad', 'Precio_sin_descuento', 'Descuento',
        'Precio_Bruto', 'Precio_Neto', 'IVA'
    ]

    for campo in campos_numericos:
        campo_normalizado = campo.replace(' ', '_').replace('(', '').replace(')', '')
        if campo_normalizado in df.columns:
            try:
                logger.info(f"   [INFO] Procesando campo numérico: {campo_normalizado}")
                df[campo_normalizado] = df[campo_normalizado].astype(str)
                df[campo_normalizado] = df[campo_normalizado].str.replace(
                    r'[^\d,.]', '', regex=True
                )
                df[campo_normalizado] = df[campo_normalizado].str.replace(
                    ',', '.', regex=False
                )
                df[campo_normalizado] = pd.to_numeric(
                    df[campo_normalizado], errors='coerce'
                ).fillna(0)
                logger.info(f"   [OK] Campo {campo_normalizado} validado")
            except Exception as e:
                logger.error(f"   [ERROR] Error validando {campo_normalizado}: {e}")

    # 7️⃣ NORMALIZACIÓN DE LA CUENTA/SEDE
    logger.info("\n7. Normalizando sedes...")
    if 'Cuenta' in df.columns:
        cuenta_lower = df['Cuenta'].astype(str).str.lower()

        conditions = [
            cuenta_lower.str.contains('plaza.bolsillo|plaza bolsillo', case=False, na=False),
            cuenta_lower.str.contains('merced', case=False, na=False),
            cuenta_lower.str.contains('tajamar', case=False, na=False),
            cuenta_lower.str.contains(
                'persa.*victor.*manuel|victor.*manuel', case=False, na=False
            )
        ]
        choices = ['Plaza Bolsillo', 'Merced', 'Tajamar', 'Persa Victor Manuel']
        df['Sede_Normalizada'] = np.select(conditions, choices, default='Sede No Identificada')

        mask_na_o_empty = df['Cuenta'].isna() | df['Cuenta'].astype(str).str.lower().isin(
            ['no_especificado', 'nan', 'none', '']
        )
        df.loc[mask_na_o_empty, 'Sede_Normalizada'] = 'Sede No Identificada'

        logger.info("   [OK] Sedes normalizadas")
        logger.info(
            f"   [INFO] Sedes identificadas: {df['Sede_Normalizada'].value_counts().to_dict()}"
        )

    # 8️⃣ FILTRADO DE DESCRIPCIONES NO DESEADAS
    logger.info("\n8. Filtrando descripciones no deseadas...")
    if 'Descripcion' in df.columns:
        filas_antes = len(df)
        df['Descripcion'] = (
            df['Descripcion']
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.strip()
            .str.lower()
        )

        mask_propinas = df['Descripcion'].str.contains(
            r'\b(tip|propina)\b',
            case=False,
            regex=True,
            na=False
        )
        df_filtrado = df[~mask_propinas]
        filas_eliminadas = filas_antes - len(df_filtrado)
        stats['filas_propinas_eliminadas'] = filas_eliminadas
        logger.info(
            f"   [OK] Descripciones de propinas filtradas: {filas_eliminadas} filas eliminadas"
        )
        df = df_filtrado
        df = registrar_etapa(df, "despues_propinas_filtradas", stats)

    # 9️⃣ VALIDACIÓN DE PRECIOS POSITIVOS
    logger.info("\n9. Validando precios positivos...")
    if 'Precio_Bruto' in df.columns:
        filas_antes = len(df)
        df['Precio_Bruto'] = pd.to_numeric(df['Precio_Bruto'], errors='coerce').fillna(0)
        mascara_validos = df['Precio_Bruto'] >= 0
        df = df[mascara_validos]
        filas_eliminadas = filas_antes - len(df)
        stats['filas_precios_negativos_eliminadas'] = filas_eliminadas
        stats['precios_validos_mayor_cero'] = (df['Precio_Bruto'] > 0).sum()
        logger.info(
            f"   [OK] Registros con precios negativos eliminados: {filas_eliminadas} filas"
        )
        df = registrar_etapa(df, "despues_negativos_eliminados", stats)

    # 🔟 VALIDACIÓN ESTRUCTURAL DE PRECIOS
    logger.info("\n10. Validando estructura de precios...")
    if all(col in df.columns for col in ['Precio_Bruto', 'Cantidad']):
        df['precio_unitario_calculado'] = np.where(
            df['Cantidad'] > 0,
            df['Precio_Bruto'] / df['Cantidad'],
            0
        )
        logger.info("   [OK] Validación de estructura completada")

    # 1️⃣1️⃣ MANEJO DE VALORES EXTREMOS
    logger.info("\n11. Detectando y tratando valores extremos...")
    if 'Precio_Bruto' in df.columns and eliminar_outliers:
        if len(df) > 10:
            Q1 = df['Precio_Bruto'].quantile(0.25)
            Q3 = df['Precio_Bruto'].quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 3 * IQR
            outliers_mask = df['Precio_Bruto'] > limite_superior
            outliers_cnt = outliers_mask.sum()
            stats['outliers_detectados'] = outliers_cnt

            if outliers_cnt > 0:
                df.loc[outliers_mask, 'Precio_Bruto'] = limite_superior
                logger.info(f"   [OK] Outliers capeados al limite superior: {limite_superior}")
                stats['outliers_capeados'] = outliers_cnt
        else:
            logger.info("   [INFO] No hay suficientes datos para detectar outliers")

    # 1️⃣2️⃣ VALIDACIÓN FINAL
    logger.info("\n12. Validación final...")
    filas_finales = len(df)
    stats['filas_finales'] = filas_finales
    stats['porcentaje_retencion'] = (
        filas_finales / stats['filas_iniciales'] * 100
        if stats['filas_iniciales'] > 0 else 0
    )
    logger.info(f"   Filas finales: {filas_finales}")
    logger.info(f"   Porcentaje de retención: {stats['porcentaje_retencion']:.2f}%")

    # 1️⃣3️⃣ EXPORTACIÓN
    if output_file:
        logger.info(f"\n13. Exportando datos limpios a {output_file}...")
        try:
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info("   [OK] Datos exportados exitosamente")
        except Exception as e:
            logger.error(f"   [ERROR] Error exportando datos: {e}")

    logger.info("=== LIMPIEZA DE INFORME_VENTAS COMPLETADA ===")
    return df, stats

# =====================================================================
# FUNCIÓN DE LIMPIEZA PARA TRANSACCIONES
# =====================================================================
def limpiar_transacciones(input_file, output_file=None):
    """
    Función específica para limpiar el archivo transacciones
    """
    logger.info("=== INICIANDO LIMPIEZA DE TRANSACCIONES ===")
    stats = {}
    
    # 1. CARGA DE DATOS
    logger.info("1. Cargando datos...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        stats['filas_iniciales'] = len(df)
        stats['columnas_iniciales'] = len(df.columns)
        logger.info(f"   ✓ Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        logger.error(f"   ✗ Error cargando archivo: {e}")
        return None, {}
    
    # 2. NORMALIZACIÓN DE TEXTOS
    logger.info("\n2. Normalizando textos...")
    columnas_texto = df.select_dtypes(include=['object']).columns.tolist()
    
    for columna in columnas_texto:
        if columna in df.columns:
            df[columna] = df[columna].astype(str)
            df[columna] = (
                df[columna]
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
                .str.strip()
                .str.lower()
            )
            df[columna] = df[columna].replace(['none', 'null', 'nan', '', 'nan'], 'no_especificado')
    
    logger.info("   ✓ Normalización de textos completada")
    
    # 3. NORMALIZACIÓN DE NOMBRES DE COLUMNAS
    logger.info("\n3. Normalizando nombres de columnas...")
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_').str.replace('á', 'a').str.replace('é', 'e').str.replace('í', 'i').str.replace('ó', 'o').str.replace('ú', 'u')
    logger.info(f"   Columnas normalizadas: {list(df.columns)}")
    
    # 4. MANEJO INTELIGENTE DE FECHAS
    logger.info("\n4. Manejo inteligente de fechas...")
    if 'Fecha' in df.columns:
        try:
            df['fecha_original_valor'] = df['Fecha'].copy()
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True)
            fechas_invalidas = df['Fecha'].isnull().sum()
            stats['fechas_invalidas'] = fechas_invalidas
            logger.info(f"   ✓ Fechas nulas detectadas: {fechas_invalidas}")
            df['fecha_problema'] = df['Fecha'].isnull()
            logger.info("   ✓ Manejo de fechas completado")
        except Exception as e:
            logger.error(f"   ✗ Error manejando fechas: {e}")
            df['fecha_problema'] = False
    
    # 5. VALIDACIÓN DE CAMPOS NUMÉRICOS
    logger.info("\n5. Validando campos numericos...")
    campos_numericos = ['Total', 'Depositos', 'Comision', 'Subtotal', 'Impuesto', 'Propina']
    
    for campo in campos_numericos:
        campo_normalizado = campo.replace(' ', '_')
        if campo_normalizado in df.columns:
            try:
                df[campo_normalizado] = pd.to_numeric(df[campo_normalizado], errors='coerce').fillna(0)
                negativos = (df[campo_normalizado] < 0).sum()
                if negativos > 0:
                    logger.warning(f"   ⚠ Valores negativos en {campo_normalizado}: {negativos}")
                    stats[f'{campo_normalizado}_negativos'] = negativos
                logger.info(f"   ✓ Campo {campo_normalizado} validado")
            except Exception as e:
                logger.error(f"   ✗ Error validando {campo_normalizado}: {e}")
    
    # 6. MANEJO DE DUPLICADOS ESPECÍFICO
    logger.info("\n6. Manejando duplicados de transacciones...")
    if 'ID_de_transaccion' in df.columns and 'Estado' in df.columns:
        filas_antes = len(df)
        duplicados_ids = df[df.duplicated('ID_de_transaccion', keep=False)]
        ids_duplicados = duplicados_ids['ID_de_transaccion'].unique()
        stats['ids_duplicados_encontrados'] = len(ids_duplicados)
        
        if len(ids_duplicados) > 0:
            logger.warning(f"   ⚠ Transacciones duplicadas encontradas: {len(ids_duplicados)} IDs")
            
            def priorizar_estado(grupo):
                prioridad = {'exitosa': 3, 'pagado': 2, 'fallida': 1, 'cancelada': 1, 'agendado': 1, 'no_especificado': 0}
                grupo['prioridad'] = grupo['Estado'].map(prioridad).fillna(0)
                max_prioridad = grupo['prioridad'].max()
                return grupo[grupo['prioridad'] == max_prioridad].iloc[0]
            
            df_unicas = df.groupby('ID_de_transaccion').apply(priorizar_estado).reset_index(drop=True)
            df_unicas = df_unicas.drop('prioridad', axis=1) if 'prioridad' in df_unicas.columns else df_unicas
            
            duplicados_eliminados = filas_antes - len(df_unicas)
            stats['duplicados_eliminados'] = duplicados_eliminados
            logger.info(f"   ✓ Duplicados eliminados: {duplicados_eliminados} filas")
            df = df_unicas
        else:
            logger.info("   ✓ No se encontraron transacciones duplicadas")
            stats['duplicados_eliminados'] = 0
    
    # 7. FILTRADO POR ESTADOS VÁLIDOS
    logger.info("\n7. Filtrando por estados válidos...")
    if 'Estado' in df.columns:
        filas_antes = len(df)
        estados_validos = ['exitosa', 'pagado']
        df_filtrado = df[df['Estado'].isin(estados_validos)]
        filas_filtradas = filas_antes - len(df_filtrado)
        stats['filas_filtradas_por_estado'] = filas_filtradas
        logger.info(f"   ✓ Estados filtrados: {filas_filtradas} filas eliminadas")
        df = df_filtrado
    
    # 8. VALIDACIÓN DE TRANSACCIONES VÁLIDAS
    logger.info("\n8. Validando transacciones válidas...")
    if 'Total' in df.columns:
        filas_antes = len(df)
        df = df[df['Total'] > 0]
        filas_eliminadas = filas_antes - len(df)
        stats['transacciones_invalidas_eliminadas'] = filas_eliminadas
        if filas_eliminadas > 0:
            logger.info(f"   ✓ Transacciones inválidas eliminadas: {filas_eliminadas} filas")
    
    # 9. VALIDACIÓN FINAL
    logger.info("\n9. Validacion final...")
    filas_finales = len(df)
    stats['filas_finales'] = filas_finales
    stats['porcentaje_retencion'] = (filas_finales / stats['filas_iniciales']) * 100 if stats['filas_iniciales'] > 0 else 0
    
    logger.info(f"   ✓ Filas finales: {filas_finales}")
    logger.info(f"   ✓ Porcentaje de retencion: {stats['porcentaje_retencion']:.2f}%")
    
    # 10. EXPORTACIÓN
    if output_file:
        logger.info(f"\n10. Exportando datos limpios a {output_file}...")
        try:
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info("   ✓ Datos exportados exitosamente")
        except Exception as e:
            logger.error(f"   ✗ Error exportando datos: {e}")
    
    logger.info("=== LIMPIEZA DE TRANSACCIONES COMPLETADA ===")
    return df, stats

# =====================================================================
# FUNCIÓN PARA UNIFICAR DATASETS - VERSIÓN CORREGIDA
# =====================================================================
def unificar_datasets_ventas(df_ventas, df_transacciones, output_file=None):
    """
    Unifica los datasets limpios para crear dataset para ML
    """
    logger.info("=== INICIANDO UNIFICACIÓN DE DATASETS ===")
    
    try:
        # Identificar columnas de ID de transacción
        ventas_id_col = 'ID_de_transacción' if 'ID_de_transacción' in df_ventas.columns else 'ID_de_transaccion'
        trans_id_col = 'ID_de_transaccion' if 'ID_de_transaccion' in df_transacciones.columns else 'ID_de_transacción'
        
        logger.info(f"   ✓ Columna ID ventas: {ventas_id_col}")
        logger.info(f"   ✓ Columna ID transacciones: {trans_id_col}")
        
        # Preparar datasets para merge
        logger.info("   ✓ Preparando datasets para merge...")
        
        # Seleccionar columnas relevantes de transacciones
        cols_trans_relevantes = [trans_id_col]
        if 'Fecha' in df_transacciones.columns:
            cols_trans_relevantes.append('Fecha')
        if 'Total' in df_transacciones.columns:
            cols_trans_relevantes.append('Total')
        if 'Subtotal' in df_transacciones.columns:
            cols_trans_relevantes.append('Subtotal')
        if 'Impuesto' in df_transacciones.columns:
            cols_trans_relevantes.append('Impuesto')
        if 'Propina' in df_transacciones.columns:
            cols_trans_relevantes.append('Propina')
        if 'Metodo_de_pago' in df_transacciones.columns:
            cols_trans_relevantes.append('Metodo_de_pago')
        if 'Estado' in df_transacciones.columns:
            cols_trans_relevantes.append('Estado')
        
        # Filtrar solo columnas que existen
        cols_trans_relevantes = [col for col in cols_trans_relevantes if col in df_transacciones.columns]
        df_trans_prep = df_transacciones[cols_trans_relevantes].copy()
        
        # Renombrar columnas para consistencia
        df_trans_prep = df_trans_prep.rename(columns={trans_id_col: 'ID_transaccion'})
        df_ventas_prep = df_ventas.copy()
        df_ventas_prep = df_ventas_prep.rename(columns={ventas_id_col: 'ID_transaccion'})
        
        # Verificar que la columna Fecha exista en transacciones
        if 'Fecha' not in df_trans_prep.columns and 'Fecha' in df_transacciones.columns:
            df_trans_prep['Fecha'] = df_transacciones['Fecha']
        
        # Merge datasets
        logger.info("   ✓ Realizando merge de datasets...")
        df_unificado = pd.merge(
            df_ventas_prep,
            df_trans_prep,
            on='ID_transaccion',
            how='left',
            suffixes=('_venta', '_trans')
        )
        
        logger.info(f"   ✓ Dataset unificado: {df_unificado.shape[0]} registros")
        
        # Crear features temporales - verificar que Fecha exista
        if 'Fecha' in df_unificado.columns:
            df_unificado['Fecha'] = pd.to_datetime(df_unificado['Fecha'], errors='coerce')
        elif 'Fecha_venta' in df_unificado.columns:
            df_unificado['Fecha'] = pd.to_datetime(df_unificado['Fecha_venta'], errors='coerce')
        elif 'Fecha_trans' in df_unificado.columns:
            df_unificado['Fecha'] = pd.to_datetime(df_unificado['Fecha_trans'], errors='coerce')
        else:
            # Usar la fecha de ventas si existe
            fecha_cols = [col for col in df_unificado.columns if 'fecha' in col.lower()]
            if fecha_cols:
                df_unificado['Fecha'] = pd.to_datetime(df_unificado[fecha_cols[0]], errors='coerce')
            else:
                logger.warning("   ⚠ No se encontró columna de fecha, creando fecha dummy")
                df_unificado['Fecha'] = pd.to_datetime('2023-01-01')
        
        # Crear componentes temporales
        df_unificado['anio'] = df_unificado['Fecha'].dt.year
        df_unificado['mes'] = df_unificado['Fecha'].dt.month
        df_unificado['dia'] = df_unificado['Fecha'].dt.day
        df_unificado['dia_semana'] = df_unificado['Fecha'].dt.dayofweek
        df_unificado['semana_anio'] = df_unificado['Fecha'].dt.isocalendar().week
        
        # Normalizar textos
        if 'Categoria' in df_unificado.columns:
            df_unificado['Categoria_normalizada'] = (
                df_unificado['Categoria']
                .astype(str)
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
                .str.strip()
                .str.lower()
            )
        else:
            df_unificado['Categoria_normalizada'] = 'no_especificado'
        
        if 'Descripcion' in df_unificado.columns:
            df_unificado['Descripcion_normalizada'] = (
                df_unificado['Descripcion']
                .astype(str)
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
                .str.strip()
                .str.lower()
            )
        else:
            df_unificado['Descripcion_normalizada'] = 'no_especificado'
        
        # Asegurar tipos numéricos
        campos_numericos = ['Cantidad', 'Precio_sin_descuento', 'Descuento', 
                           'Precio_Bruto', 'Precio_Neto', 'IVA', 'Total']
        
        for campo in campos_numericos:
            if campo in df_unificado.columns:
                df_unificado[campo] = pd.to_numeric(df_unificado[campo], errors='coerce').fillna(0)
        
        # Crear dataset agregado para ML
        logger.info("   ✓ Creando dataset agregado para ML...")
        
        # Definir columnas de agrupación disponibles
        cols_agrupacion = ['Descripcion_normalizada', 'Categoria_normalizada']
        if 'Sede_Normalizada' in df_unificado.columns:
            cols_agrupacion.append('Sede_Normalizada')
        cols_agrupacion.extend(['anio', 'mes', 'dia', 'dia_semana'])
        
        # Filtrar solo columnas que existen
        cols_agrupacion = [col for col in cols_agrupacion if col in df_unificado.columns]
        
        # Agregar verificación de columnas requeridas
        required_cols = ['Cantidad']
        if not all(col in df_unificado.columns for col in required_cols):
            logger.error(f"   ✗ Columnas requeridas no encontradas: {[col for col in required_cols if col not in df_unificado.columns]}")
            return None
        
        # Agregación
        agg_dict = {'Cantidad': 'sum'}
        if 'Precio_Neto' in df_unificado.columns:
            agg_dict['Precio_Neto'] = 'mean'
        if 'Total' in df_unificado.columns:
            agg_dict['Total'] = 'sum'
        agg_dict['ID_transaccion'] = 'nunique'
        
        df_ml = df_unificado.groupby(cols_agrupacion).agg(agg_dict).reset_index()
        
        # Renombrar columnas para claridad
        rename_dict = {
            'Cantidad': 'demanda_total',
            'ID_transaccion': 'num_transacciones'
        }
        if 'Precio_Neto' in df_unificado.columns:
            rename_dict['Precio_Neto'] = 'precio_promedio'
        if 'Total' in df_unificado.columns:
            rename_dict['Total'] = 'venta_total'
            
        df_ml = df_ml.rename(columns=rename_dict)
        
        # Agregar features adicionales solo si las columnas existen
        if 'Descripcion_normalizada' in df_ml.columns and 'Sede_Normalizada' in df_ml.columns:
            try:
                demanda_producto_sede = df_ml.groupby(['Descripcion_normalizada', 'Sede_Normalizada'])['demanda_total'].mean().reset_index()
                demanda_producto_sede = demanda_producto_sede.rename(columns={'demanda_total': 'demanda_promedio_producto'})
                df_ml = pd.merge(df_ml, demanda_producto_sede, on=['Descripcion_normalizada', 'Sede_Normalizada'], how='left')
            except Exception as e:
                logger.warning(f"   ⚠ Error calculando demanda promedio: {e}")
        
        # Validación final
        if 'demanda_total' in df_ml.columns and 'Sede_Normalizada' in df_ml.columns:
            df_ml = df_ml[
                (df_ml['demanda_total'] >= 0) & 
                (df_ml['Sede_Normalizada'] != 'Sede No Identificada')
            ]
        
        logger.info(f"   ✓ Dataset ML final: {df_ml.shape[0]} registros, {df_ml.shape[1]} columnas")
        
        # Exportar
        if output_file:
            df_ml.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"   ✓ Dataset ML guardado en: {output_file}")
            
            # Guardar también el dataset detallado
            detalle_file = output_file.replace('.csv', '_detalle.csv')
            df_unificado.to_csv(detalle_file, index=False, encoding='utf-8')
            logger.info(f"   ✓ Dataset detallado guardado en: {detalle_file}")
        
        logger.info("=== UNIFICACIÓN COMPLETADA ===")
        return df_ml
        
    except Exception as e:
        logger.error(f"   ✗ Error en unificación: {str(e)}")
        return None

# =====================================================================
# FUNCIÓN PRINCIPAL - EJECUTAR TODO EL PROCESO
# =====================================================================
def ejecutar_etl_completo():
    """
    Ejecuta todo el proceso ETL completo
    """
    logger.info("🚀 INICIANDO PROCESO ETL COMPLETO")
    
    # Rutas de archivos
    ruta_ventas = r"C:\Users\Francisco\Downloads\ETL PORTACAFE\informe_ventas.csv"
    ruta_transacciones = r"C:\Users\Francisco\Downloads\ETL PORTACAFE\transacciones.csv"
    
    # Verificar que existan los archivos
    if not os.path.exists(ruta_ventas):
        logger.error(f"❌ No se encuentra archivo de ventas: {ruta_ventas}")
        return None
    
    if not os.path.exists(ruta_transacciones):
        logger.error(f"❌ No se encuentra archivo de transacciones: {ruta_transacciones}")
        return None
    
    logger.info("✅ Archivos encontrados")
    
    try:
        # 1. LIMPIAR INFORME DE VENTAS
        logger.info("📋 LIMPIANDO INFORME DE VENTAS...")
        df_ventas_limpio, stats_ventas = limpiar_informe_ventas(
            ruta_ventas, 
            output_file='informe_ventas_limpio.csv'
        )
        
        if df_ventas_limpio is None:
            logger.error("❌ Error limpiando informe de ventas")
            return None
        
        logger.info(f"✅ Informe de ventas limpio: {len(df_ventas_limpio)} registros")
        
        # 2. LIMPIAR TRANSACCIONES
        logger.info("💳 LIMPIANDO TRANSACCIONES...")
        df_trans_limpio, stats_trans = limpiar_transacciones(
            ruta_transacciones,
            output_file='transacciones_limpio.csv'
        )
        
        if df_trans_limpio is None:
            logger.error("❌ Error limpiando transacciones")
            return None
        
        logger.info(f"✅ Transacciones limpias: {len(df_trans_limpio)} registros")
        
        # 3. UNIFICAR DATASETS
        logger.info("🔗 UNIFICANDO DATASETS...")
        df_ml = unificar_datasets_ventas(
            df_ventas_limpio,
            df_trans_limpio,
            output_file='dataset_ml_prediccion.csv'
        )
        
        if df_ml is None:
            logger.error("❌ Error unificando datasets")
            return None
        
        # 4. GENERAR REPORTES
        logger.info("📊 GENERANDO REPORTES...")
        
        # Reporte de ventas
        reporte_ventas = {
            'total_registros_iniciales': stats_ventas.get('filas_iniciales', 0),
            'total_registros_finales': stats_ventas.get('filas_finales', 0),
            'porcentaje_retencion': stats_ventas.get('porcentaje_retencion', 0),
            'sedes_identificadas': df_ventas_limpio['Sede_Normalizada'].value_counts().to_dict() if 'Sede_Normalizada' in df_ventas_limpio.columns else {}
        }
        
        with open('reporte_ventas.json', 'w', encoding='utf-8') as f:
            json.dump(reporte_ventas, f, indent=2, default=str, ensure_ascii=False)
        
        # Reporte de transacciones
        reporte_trans = {
            'total_registros_iniciales': stats_trans.get('filas_iniciales', 0),
            'total_registros_finales': stats_trans.get('filas_finales', 0),
            'porcentaje_retencion': stats_trans.get('porcentaje_retencion', 0),
            'duplicados_eliminados': stats_trans.get('duplicados_eliminados', 0)
        }
        
        with open('reporte_transacciones.json', 'w', encoding='utf-8') as f:
            json.dump(reporte_trans, f, indent=2, default=str, ensure_ascii=False)
        
        # Reporte del dataset ML
        if df_ml is not None:
            reporte_ml = {
                'total_registros': len(df_ml),
                'total_columnas': len(df_ml.columns),
                'columnas': list(df_ml.columns),
                'sedes': df_ml['Sede_Normalizada'].value_counts().to_dict() if 'Sede_Normalizada' in df_ml.columns else {},
                'productos_unicos': df_ml['Descripcion_normalizada'].nunique() if 'Descripcion_normalizada' in df_ml.columns else 0,
                'categorias': df_ml['Categoria_normalizada'].value_counts().to_dict() if 'Categoria_normalizada' in df_ml.columns else {},
            }
            
            with open('reporte_dataset_ml.json', 'w', encoding='utf-8') as f:
                json.dump(reporte_ml, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info("✅ REPORTES GENERADOS")
        
        # 5. MOSTRAR RESUMEN FINAL
        logger.info(" PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 50)
        logger.info("ARCHIVOS GENERADOS:")
        logger.info("├── informe_ventas_limpio.csv")
        logger.info("├── transacciones_limpio.csv") 
        logger.info("├── dataset_ml_prediccion.csv")
        logger.info("├── dataset_ml_prediccion_detalle.csv")
        logger.info("├── reporte_ventas.json")
        logger.info("├── reporte_transacciones.json")
        logger.info("└── reporte_dataset_ml.json")
        logger.info("=" * 50)
        if df_ml is not None:
            logger.info(f" DATASET ML FINAL:")
            logger.info(f"   • Registros: {len(df_ml):,}")
            logger.info(f"   • Columnas: {len(df_ml.columns)}")
            if 'Descripcion_normalizada' in df_ml.columns:
                logger.info(f"   • Productos únicos: {df_ml['Descripcion_normalizada'].nunique():,}")
            if 'Sede_Normalizada' in df_ml.columns:
                logger.info(f"   • Sedes: {df_ml['Sede_Normalizada'].nunique()}")
        
        return df_ml
        
    except Exception as e:
        logger.error(f"❌ Error en proceso ETL: {str(e)}")
        return None

# =====================================================================
# EJECUTAR EL PROCESO
# =====================================================================
if __name__ == "__main__":
    dataset_final = ejecutar_etl_completo()
    
    if dataset_final is not None:
        logger.info("✅ PROCESO ETL FINALIZADO CORRECTAMENTE")
        logger.info("📁 Los archivos han sido generados en el directorio actual")
    else:
        logger.error("❌ EL PROCESO ETL NO SE COMPLETÓ")
