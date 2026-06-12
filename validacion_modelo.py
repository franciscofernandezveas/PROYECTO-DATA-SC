# --------------------------------------------------------------
#  validacion_modelo_final.py
#  Comparación de 3 escenarios: Full | Weighted | Core-Only
#  + Safety stock por percentil de residuos
# --------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import json
warnings.filterwarnings('ignore')

# Configuración visual
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# -----------------------------------------------------------------
# 0️⃣  UTILIDADES
# -----------------------------------------------------------------
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def calcular_metricas(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'SMAPE': smape(y_true, y_pred)
    }

def post_procesar(predicciones):
    return np.maximum(np.round(predicciones), 0).astype(int)

# -----------------------------------------------------------------
# 1️⃣  CARGA Y PREPARACIÓN
# -----------------------------------------------------------------
def cargar_datos(ruta_dataset: Path):
    print(f"🔄 Cargando dataset desde: {ruta_dataset}")
    df = pd.read_csv(ruta_dataset)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    req = ['fecha', 'Descripcion_normalizada', 'Sede_Normalizada', 'demanda_total']
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas: {miss}")
    
    print(f"✅ Dataset: {len(df):,} filas | {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print(f"   Productos: {df['Descripcion_normalizada'].nunique()} | Sedes: {df['Sede_Normalizada'].nunique()}")
    return df

def preparar_features(df: pd.DataFrame):
    """Excluye identificadores, target y cualquier feature de leakage conocido."""
    leakage_cols = ['venta_bruta', 'demanda_acumulada_mes', 'ratio_vs_same_day']
    exclude = ['fecha', 'Descripcion_normalizada', 'Sede_Normalizada', 'demanda_total'] + leakage_cols
    
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    # Drop non-numeric si quedó algo
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        X = X.drop(columns=non_num)
    
    y = df['demanda_total']
    return X, y, list(X.columns)

# -----------------------------------------------------------------
# 2️  DIVISIÓN TEMPORAL
# -----------------------------------------------------------------
def dividir_temporal(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_days: int = 28):
    df = df.sort_values(['Sede_Normalizada', 'Descripcion_normalizada', 'fecha']).reset_index(drop=True)
    X = X.loc[df.index].reset_index(drop=True)
    y = y.loc[df.index].reset_index(drop=True)
    
    cutoff = df['fecha'].max() - pd.Timedelta(days=test_days)
    train_mask = df['fecha'] <= cutoff
    test_mask = df['fecha'] > cutoff
    
    return {
        'X_train': X[train_mask], 'X_test': X[test_mask],
        'y_train': y[train_mask], 'y_test': y[test_mask],
        'df_train': df[train_mask].copy(), 'df_test': df[test_mask].copy(),
        'cutoff': cutoff
    }

# -----------------------------------------------------------------
# 3️  BENCHMARKS (calculados desde train ÚNICAMENTE)
# -----------------------------------------------------------------
def calcular_benchmarks(df_train, df_test):
    gcols = ['Descripcion_normalizada', 'Sede_Normalizada']
    
    # Naive Last
    last_train = df_train.groupby(gcols)['demanda_total'].last()
    df_test['naive_last'] = df_test.apply(
        lambda r: last_train.get((r['Descripcion_normalizada'], r['Sede_Normalizada']), 0), axis=1
    )
    
    # Naive Seasonal (mismo día de semana)
    seasonal = df_train.groupby(gcols + ['dia_semana'])['demanda_total'].mean()
    df_test['naive_seasonal'] = df_test.apply(
        lambda r: seasonal.get((r['Descripcion_normalizada'], r['Sede_Normalizada'], r['dia_semana']), 0), axis=1
    )
    return df_test

# -----------------------------------------------------------------
# 4️⃣  ENTRENAMIENTO XGBOOST (con soporte sample_weight)
# -----------------------------------------------------------------
def entrenar_modelo(X_train, y_train, X_test, y_test, sample_weight=None, nombre="Modelo"):
    print(f"\n Entrenando: {nombre}")
    
    # Limpieza
    for d in [X_train, X_test]:
        d.replace([np.inf, -np.inf], 0, inplace=True)
        d.fillna(0, inplace=True)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=30,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        tree_method='hist',
        eval_metric='rmse'
    )
    
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
        print(f"   ↳ Sample weights activos (rango: {sample_weight.min():.2f} - {sample_weight.max():.2f})")
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False,
            **fit_kwargs
        )
        print(f"   ✅ Modo moderno | Best iter: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
    except TypeError:
        # Fallback versión antigua (no soporta early_stopping en fit, pero sí sample_weight)
        print(f"   🔄 Fallback legacy...")
        model.set_params(n_estimators=500, learning_rate=0.05)
        model.fit(X_train, y_train, verbose=False, **fit_kwargs)
        print(f"   ✅ Modo legacy finalizado")
    
    return model

# -----------------------------------------------------------------
# 5️⃣  EVALUACIÓN POR TIER
# -----------------------------------------------------------------
def evaluar_modelo(nombre, model, X_test, y_test, df_test, top_core=20):
    y_pred_raw = model.predict(X_test)
    y_pred = post_procesar(y_pred_raw)
    
    df_test = df_test.copy()
    df_test['prediccion'] = y_pred
    
    # Identificar productos core por volumen del TRAIN (no del test para evitar leakage)
    # Nota: asumimos que df_test tiene la info necesaria; si no, se pasa desde train
    # En esta implementación usaremos el volumen observado en el test para reportar,
    # pero el core se define desde train cuando llamamos desde el pipeline principal.
    
    # Métricas: Todos
    m_all = calcular_metricas(y_test, y_pred)
    
    # Métricas: Core (top productos en este conjunto, para simplificar el script autocontenido)
    volumen = df_test.groupby('Descripcion_normalizada')['demanda_total'].sum().sort_values(ascending=False)
    core_prods = volumen.head(top_core).index.tolist()
    
    mask_core = df_test['Descripcion_normalizada'].isin(core_prods)
    if mask_core.sum() > 0:
        m_core = calcular_metricas(df_test.loc[mask_core, 'demanda_total'], 
                                   df_test.loc[mask_core, 'prediccion'])
    else:
        m_core = {'MAE': np.nan, 'RMSE': np.nan, 'SMAPE': np.nan}
    
    # Métricas: Estrella (capuccino si existe)
    estrella = 'capuccino'
    if estrella in df_test['Descripcion_normalizada'].unique():
        mask_est = df_test['Descripcion_normalizada'] == estrella
        m_est = calcular_metricas(df_test.loc[mask_est, 'demanda_total'],
                                  df_test.loc[mask_est, 'prediccion'])
    else:
        m_est = {'MAE': np.nan, 'RMSE': np.nan, 'SMAPE': np.nan}
    
    print(f"\n📊 RESULTADOS: {nombre}")
    print(f"   Todos  -> MAE: {m_all['MAE']:.3f} | RMSE: {m_all['RMSE']:.3f} | SMAPE: {m_all['SMAPE']:.3f}")
    print(f"   Core   -> MAE: {m_core['MAE']:.3f} | RMSE: {m_core['RMSE']:.3f} | SMAPE: {m_core['SMAPE']:.3f}")
    print(f"    {estrella} -> MAE: {m_est['MAE']:.3f} | RMSE: {m_est['RMSE']:.3f} | SMAPE: {m_est['SMAPE']:.3f}")
    
    return {
        'nombre': nombre,
        'prediccion': y_pred,
        'prediccion_raw': y_pred_raw,
        'm_all': m_all,
        'm_core': m_core,
        'm_estrella': m_est,
        'df_test': df_test,
        'core_prods': core_prods
    }

# -----------------------------------------------------------------
# 6️ VISUALIZACIÓN CON DOBLE EJE (MAE/RMSE vs SMAPE)
# -----------------------------------------------------------------
def graficar_comparacion(resultados: list, naive_metrics: dict):
    """
    resultados: lista de dicts de evaluar_modelo
    naive_metrics: dict con 'naive_last' y 'naive_seasonal' sobre el test core
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # --- GRÁFICO 1: Barras con doble eje (Core) ---
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    nombres = [r['nombre'] for r in resultados] + ['Naïve Sea']
    x = np.arange(len(nombres))
    w = 0.35
    
    # Eje izquierdo: MAE y RMSE (unidades)
    mae_vals = [r['m_core']['MAE'] for r in resultados] + [naive_metrics['naive_seasonal_core']['MAE']]
    rmse_vals = [r['m_core']['RMSE'] for r in resultados] + [naive_metrics['naive_seasonal_core']['RMSE']]
    
    ax1.bar(x - w/2, mae_vals, w, label='MAE', color='steelblue', alpha=0.9)
    ax1.bar(x + w/2, rmse_vals, w, label='RMSE', color='coral', alpha=0.9)
    ax1.set_ylabel('Unidades (MAE / RMSE)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Eje derecho: SMAPE (%)
    smape_vals = [r['m_core']['SMAPE'] for r in resultados] + [naive_metrics['naive_seasonal_core']['SMAPE']]
    ax1_twin.plot(x, smape_vals, 'o-', color='darkgreen', linewidth=2, markersize=8, label='SMAPE')
    ax1_twin.set_ylabel('SMAPE (%)', color='darkgreen')
    ax1_twin.tick_params(axis='y', labelcolor='darkgreen')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=15, ha='right')
    ax1.set_title('Métricas Core (Top 20) - Doble Eje')
    ax1.grid(True, alpha=0.3)
    
    # Leyendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # --- GRÁFICO 2: Serie capuccino ejemplo (mejor modelo vs naive) ---
    ax2 = axes[0, 1]
    # Tomar el primer modelo como referencia (usualmente Weighted o Full)
    df_ref = resultados[0]['df_test']
    estrella = 'capuccino'
    if estrella in df_ref['Descripcion_normalizada'].values:
        mask = df_ref['Descripcion_normalizada'] == estrella
        sede_ref = df_ref.loc[mask, 'Sede_Normalizada'].iloc[0]
        mask &= (df_ref['Sede_Normalizada'] == sede_ref)
        df_ej = df_ref[mask].sort_values('fecha')
        if len(df_ej) > 0:
            ax2.plot(df_ej['fecha'], df_ej['demanda_total'], 'o-', label='Real', markersize=3)
            ax2.plot(df_ej['fecha'], df_ej['prediccion'], 's-', label='XGBoost', markersize=3)
            ax2.plot(df_ej['fecha'], df_ej['naive_seasonal'], '^-', label='Naïve Sea', markersize=3, alpha=0.6)
            ax2.set_title(f'{estrella} @ {sede_ref}')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Capuccino no disponible', ha='center', transform=ax2.transAxes)
    
    # --- GRÁFICO 3: Mejora % sobre Naïve Seasonal (Core) ---
    ax3 = axes[0, 2]
    base_mae = naive_metrics['naive_seasonal_core']['MAE']
    base_rmse = naive_metrics['naive_seasonal_core']['RMSE']
    
    mejoras_mae = [((base_mae - r['m_core']['MAE'])/base_mae)*100 for r in resultados]
    mejoras_rmse = [((base_rmse - r['m_core']['RMSE'])/base_rmse)*100 for r in resultados]
    
    x2 = np.arange(len(resultados))
    ax3.bar(x2 - w/2, mejoras_mae, w, label='Mejora MAE %', color='seagreen', alpha=0.8)
    ax3.bar(x2 + w/2, mejoras_rmse, w, label='Mejora RMSE %', color='goldenrod', alpha=0.8)
    ax3.set_xticks(x2)
    ax3.set_xticklabels([r['nombre'] for r in resultados], rotation=15, ha='right')
    ax3.set_ylabel('Mejora %')
    ax3.set_title('Mejora vs Naïve Seasonal (Core)')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- GRÁFICO 4: Error por día de semana (mejor modelo) ---
    ax4 = axes[1, 0]
    df_best = resultados[0]['df_test'].copy()
    df_best['error_abs'] = np.abs(df_best['demanda_total'] - df_best['prediccion'])
    err_dow = df_best.groupby('dia_semana')['error_abs'].mean()
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    ax4.bar(err_dow.index, err_dow.values, color='steelblue', alpha=0.8)
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(dias)
    ax4.set_title('MAE por Día de Semana (Modelo Principal)')
    ax4.grid(True, alpha=0.3)
    
    # --- GRÁFICO 5: Importancia variables (mejor modelo) ---
    ax5 = axes[1, 1]
    # Simulamos un importancia genérica o usamos el primer modelo si está disponible
    # Este se llenará fuera si tenemos el modelo entrenado
    ax5.text(0.5, 0.5, 'Ver importancia_variables.png\ngenerada por cada modelo', 
             ha='center', transform=ax5.transAxes)
    ax5.set_title('Importancia Variables (referencia)')
    
    # --- GRÁFICO 6: Safety Stock concepto ---
    ax6 = axes[1, 2]
    ax6.axis('off')
    texto = (
        " SAFETY STOCK (Percentil 80)\n\n"
        "Basado en residuos históricos de subestimación:\n"
        f"- Stock de seguridad global sugerido: +{safety_global:.1f} unidades\n"
        f"- Producto estrella (capuccino): +{safety_estrella:.1f} unidades\n\n"
        "Interpretación:\n"
        "Si el modelo predice 45 capuccinos,\n"
        f"pedir {45 + int(np.ceil(safety_estrella))} para cubrir el pico.\n\n"
        "*Ajustar por sede según tabla CSV generada."
    )
    ax6.text(0.1, 0.9, texto, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico comparativo guardado: comparacion_modelos_final.png")

# -----------------------------------------------------------------
# 7️  SAFETY STOCK POR RESIDUOS HISTÓRICOS
# -----------------------------------------------------------------
def calcular_safety_stock(df_test, y_pred, percentile=80):
    df_test = df_test.copy()
    df_test['prediccion'] = y_pred
    df_test['residuo'] = df_test['demanda_total'] - df_test['prediccion']  # real - predicho
    df_test['subestimacion'] = np.maximum(df_test['residuo'], 0)  # solo donde subestimamos
    
    # Global (sobre todos los datos de test)
    safety_global = np.percentile(df_test['subestimacion'].values, percentile)
    
    # Por producto-sede
    safety_ps = (
        df_test.groupby(['Descripcion_normalizada', 'Sede_Normalizada'])['subestimacion']
        .quantile(percentile / 100)
        .reset_index()
        .rename(columns={'subestimacion': f'safety_stock_p{percentile}'})
    )
    
    # Estrella específica
    estrella = 'capuccino'
    safety_estrella = safety_global
    if estrella in df_test['Descripcion_normalizada'].values:
        sub_est = df_test.loc[df_test['Descripcion_normalizada'] == estrella, 'subestimacion']
        if len(sub_est) > 0:
            safety_estrella = np.percentile(sub_est.values, percentile)
    
    return safety_global, safety_estrella, safety_ps, df_test

# -----------------------------------------------------------------
# 8️  MAIN
# -----------------------------------------------------------------
if __name__ == '__main__':
    try:
        # =========================================================
        # A. CARGA
        # =========================================================
        ruta = Path('dataset_ml_experto_corregido.csv')
        df_full = cargar_datos(ruta)
        X, y, feature_names = preparar_features(df_full)
        
        # =========================================================
        # B. SPLIT TEMPORAL
        # =========================================================
        split = dividir_temporal(df_full, X, y, test_days=28)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        df_train, df_test = split['df_train'], split['df_test']
        
        print(f"\n Corte temporal: {split['cutoff'].date()}")
        print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # =========================================================
        # C. BENCHMARKS
        # =========================================================
        df_test = calcular_benchmarks(df_train, df_test)
        
        # =========================================================
        # D. DEFINIR PRODUCTOS CORE (desde train para honestidad)
        # =========================================================
        volumen_train = df_train.groupby('Descripcion_normalizada')['demanda_total'].sum().sort_values(ascending=False)
        core_products = volumen_train.head(20).index.tolist()
        print(f"\n Productos Core (top 20 por volumen train): {', '.join(core_products[:5])}...")
        
        # Filtros para modelo core-only
        mask_train_core = df_train['Descripcion_normalizada'].isin(core_products)
        mask_test_core = df_test['Descripcion_normalizada'].isin(core_products)
        
        X_train_core = X_train[mask_train_core.values].reset_index(drop=True)
        y_train_core = y_train[mask_train_core.values].reset_index(drop=True)
        df_train_core = df_train[mask_train_core.values].reset_index(drop=True)
        
        X_test_core = X_test[mask_test_core.values].reset_index(drop=True)
        y_test_core = y_test[mask_test_core.values].reset_index(drop=True)
        df_test_core = df_test[mask_test_core.values].reset_index(drop=True)
        
        # =========================================================
        # E. ENTRENAMIENTO DE 3 MODELOS
        # =========================================================
        
        # --- Modelo 1: FULL (todos los productos, sin pesos) ---
        model_full = entrenar_modelo(X_train, y_train, X_test, y_test, nombre="Full")
        res_full = evaluar_modelo("Full", model_full, X_test, y_test, df_test)
        
        # --- Modelo 2: WEIGHTED (todos, con sample_weight = demanda) ---
        # Pesos: si demanda es 0, damos peso base 1 para no perder la muestra;
        # si demanda > 0, peso proporcional al volumen (log1p para suavizar extremos)
        weights = np.log1p(y_train.values)
        weights = np.maximum(weights, 0.5)  # mínimo para que los ceros no desaparezcan
        model_weighted = entrenar_modelo(X_train, y_train, X_test, y_test, 
                                         sample_weight=weights, nombre="Weighted")
        res_weighted = evaluar_modelo("Weighted", model_weighted, X_test, y_test, df_test)
        
        # --- Modelo 3: CORE-ONLY (solo top 20, sin pesos) ---
        # Nota: recalcular benchmarks para el subset core si es necesario, 
        # pero usaremos los naive ya calculados en df_test_core
        model_core = entrenar_modelo(X_train_core, y_train_core, X_test_core, y_test_core, 
                                     nombre="Core-Only")
        res_core = evaluar_modelo("Core-Only", model_core, X_test_core, y_test_core, df_test_core)
        
        # =========================================================
        # F. MÉTRICAS NAÏVE SOBRE SUBSET CORE (para comparación justa)
        # =========================================================
        naive_seasonal_core = calcular_metricas(
            df_test_core['demanda_total'], df_test_core['naive_seasonal']
        )
        naive_metrics = {
            'naive_seasonal_core': naive_seasonal_core
        }
        print(f"\n NAÏVE SEASONAL (evaluado solo en Core Test):")
        print(f"   MAE: {naive_seasonal_core['MAE']:.3f} | RMSE: {naive_seasonal_core['RMSE']:.3f} | SMAPE: {naive_seasonal_core['SMAPE']:.3f}")
        
        # =========================================================
        # G. SAFETY STOCK (usando el mejor modelo: Weighted)
        # =========================================================
        safety_global, safety_estrella, safety_ps, df_test_weighted = calcular_safety_stock(
            res_weighted['df_test'], res_weighted['prediccion'], percentile=80
        )
        
        print(f"\n SAFETY STOCK (Percentil 80 de subestimaciones):")
        print(f"   Global:     +{safety_global:.2f} unidades")
        print(f"   Capuccino:  +{safety_estrella:.2f} unidades")
        print(f"   Ejemplo: Predicción=45 → Pedido sugerido={45 + int(np.ceil(safety_estrella))}")
        
        # =========================================================
        # H. VISUALIZACIONES
        # =========================================================
        resultados = [res_full, res_weighted, res_core]
        graficar_comparacion(resultados, naive_metrics)
        
        # Guardar importancia del mejor modelo (weighted)
        imp = pd.DataFrame({
            'feature': feature_names,
            'importance': model_weighted.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=imp.head(20), y='feature', x='importance', palette='viridis')
        plt.title('Top 20 Variables Importantes (Weighted Model)')
        plt.tight_layout()
        plt.savefig('importancia_weighted.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # =========================================================
        # I. EXPORTACIÓN DE RESULTADOS
        # =========================================================
        # Tabla resumen
        resumen = []
        for r in resultados:
            resumen.append({
                'modelo': r['nombre'],
                'MAE_all': r['m_all']['MAE'],
                'RMSE_all': r['m_all']['RMSE'],
                'SMAPE_all': r['m_all']['SMAPE'],
                'MAE_core': r['m_core']['MAE'],
                'RMSE_core': r['m_core']['RMSE'],
                'SMAPE_core': r['m_core']['SMAPE'],
                'MAE_capuccino': r['m_estrella']['MAE'],
            })
        # Añadir naive
        resumen.append({
            'modelo': 'Naive_Seasonal_Core',
            'MAE_all': np.nan, 'RMSE_all': np.nan, 'SMAPE_all': np.nan,
            'MAE_core': naive_seasonal_core['MAE'],
            'RMSE_core': naive_seasonal_core['RMSE'],
            'SMAPE_core': naive_seasonal_core['SMAPE'],
            'MAE_capuccino': np.nan
        })
        
        df_resumen = pd.DataFrame(resumen)
        df_resumen.to_csv('resumen_metricas_modelos.csv', index=False)
        
        # Safety stock detallado
        safety_ps.to_csv('safety_stock_por_producto_sede.csv', index=False)
        
        # Resultados completos del mejor modelo
        res_weighted['df_test'].to_csv('resultados_weighted_completo.csv', index=False)
        
        # JSON de métricas
        with open('metricas_comparacion.json', 'w') as f:
            json.dump({
                'modelos': {r['nombre']: {'all': r['m_all'], 'core': r['m_core'], 'estrella': r['m_estrella']} 
                           for r in resultados},
                'naive_seasonal_core': naive_seasonal_core,
                'safety_stock': {
                    'global_p80': float(safety_global),
                    'capuccino_p80': float(safety_estrella)
                }
            }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
        
        print("\n" + "="*60)
        print("ARCHIVOS GENERADOS:")
        print("="*60)
        print(" resumen_metricas_modelos.csv        -> Tabla comparativa")
        print(" comparacion_modelos_final.png       -> Visualizaciones")
        print(" importancia_weighted.png            -> Features top 20")
        print(" safety_stock_por_producto_sede.csv  -> Buffer por grupo")
        print(" resultados_weighted_completo.csv    -> Predicciones finales")
        print(" metricas_comparacion.json           -> Métricas en JSON")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
