import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyodbc
import pandas as pd
from flask import render_template, jsonify, request

from config.config import CONNECTION_STRING
from .prediction import api_predict_logic


def get_conn():
    return pyodbc.connect(CONNECTION_STRING)


def register_routes(app):

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/provinces')
    def api_provinces():
        try:
            conn = get_conn()
            df = pd.read_sql("""
                SELECT DISTINCT province_id, province_name FROM DIM_STORE
                WHERE province_name IS NOT NULL ORDER BY province_name
            """, conn)
            conn.close()
            return jsonify(df.fillna('').to_dict('records'))
        except Exception:
            return jsonify([]), 200

    @app.route('/api/stores')
    def api_stores():
        province_id = request.args.get('province_id')
        try:
            conn = get_conn()
            if province_id:
                df = pd.read_sql("""
                    SELECT store_id, store_name FROM DIM_STORE
                    WHERE province_id = ? ORDER BY store_name
                """, conn, params=[province_id])
            else:
                df = pd.read_sql(
                    "SELECT store_id, store_name FROM DIM_STORE ORDER BY store_name", conn)
            conn.close()
            return jsonify(df.fillna('').to_dict('records'))
        except Exception:
            return jsonify([])

    @app.route('/api/maingroups')
    def api_maingroups():
        try:
            conn = get_conn()
            df = pd.read_sql("""
                SELECT DISTINCT dp.maingroup_id, dp.maingroup_name
                FROM DIM_PRODUCT dp
                WHERE dp.maingroup_id IN (
                    SELECT DISTINCT dp2.maingroup_id
                    FROM FACT_DAILY_SALES f
                    JOIN DIM_PRODUCT dp2 ON dp2.product_key = f.product_key
                )
                AND dp.maingroup_name IS NOT NULL
                AND dp.maingroup_name NOT LIKE '%[0-9]%'
                ORDER BY dp.maingroup_name
            """, conn)
            conn.close()
            return jsonify(df.fillna('').to_dict('records'))
        except Exception:
            return jsonify([])

    @app.route('/api/subgroups')
    def api_subgroups():
        maingroup_id = request.args.get('maingroup_id')
        try:
            conn = get_conn()
            if maingroup_id:
                df = pd.read_sql("""
                    SELECT DISTINCT dp.subgroup_id, dp.subgroup_name
                    FROM DIM_PRODUCT dp
                    WHERE dp.maingroup_id = ?
                    AND dp.subgroup_id IN (
                        SELECT DISTINCT dp2.subgroup_id
                        FROM FACT_DAILY_SALES f
                        JOIN DIM_PRODUCT dp2 ON dp2.product_key = f.product_key
                    )
                    AND dp.subgroup_name IS NOT NULL
                    AND dp.subgroup_name NOT LIKE '%[0-9]%'
                    ORDER BY dp.subgroup_name
                """, conn, params=[maingroup_id])
            else:
                df = pd.read_sql("""
                    SELECT DISTINCT dp.subgroup_id, dp.subgroup_name
                    FROM DIM_PRODUCT dp
                    WHERE dp.subgroup_id IN (
                        SELECT DISTINCT dp2.subgroup_id
                        FROM FACT_DAILY_SALES f
                        JOIN DIM_PRODUCT dp2 ON dp2.product_key = f.product_key
                    )
                    AND dp.subgroup_name IS NOT NULL
                    AND dp.subgroup_name NOT LIKE '%[0-9]%'
                    ORDER BY dp.subgroup_name
                """, conn)
            conn.close()
            return jsonify(df.fillna('').to_dict('records'))
        except Exception:
            return jsonify([])

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        return api_predict_logic(request.json)
