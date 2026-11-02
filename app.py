import hashlib
import os
import pickle
import re
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

DB_PATH = 'vehicle_app.db'
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'vehicle_app')
DB_BACKEND = os.getenv('DB_BACKEND', 'mysql' if os.getenv('MYSQL_PASSWORD') else 'sqlite').lower()
APPOINTMENT_TIMES = ['09:00 AM', '10:00 AM', '11:00 AM', '01:00 PM', '02:00 PM', '03:00 PM']
BOOKING_STATUSES = ['Pending', 'Confirmed', 'Completed', 'Cancelled']
MODEL_CANDIDATES = ['hhmodel.pkl', 'model.pkl']
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), 'hhmodel.pkl')
FEATURE_COLUMNS = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp',
    'Temperature_difference',
]

custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}


try:
    import mysql.connector as mysql_connector
except ImportError:
    mysql_connector = None


class MySQLConnectionAdapter:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, statement, params=()):
        if statement.strip().upper() == 'BEGIN IMMEDIATE':
            statement = 'START TRANSACTION'
        else:
            statement = statement.replace('?', '%s')
        cursor = self.connection.cursor()
        cursor.execute(statement, params)
        return cursor

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


def get_db_connection():
    if DB_BACKEND == 'sqlite':
        return sqlite3.connect(DB_PATH)
    if DB_BACKEND != 'mysql':
        raise ValueError("DB_BACKEND must be either 'sqlite' or 'mysql'.")
    if mysql_connector is None:
        raise RuntimeError('MySQL backend selected. Install dependencies from requirements.txt first.')

    password = os.getenv('MYSQL_PASSWORD')
    if not password:
        raise RuntimeError('Set MYSQL_PASSWORD in the environment before starting the app with MySQL.')
    connection = mysql_connector.connect(
        host=os.getenv('MYSQL_HOST', '127.0.0.1'),
        port=int(os.getenv('MYSQL_PORT', '3306')),
        user=os.getenv('MYSQL_USER', 'root'),
        password=password,
        database=MYSQL_DATABASE,
    )
    return MySQLConnectionAdapter(connection)


def is_integrity_error(error):
    if isinstance(error, sqlite3.IntegrityError):
        return True
    return getattr(error, 'errno', None) in {1048, 1062, 1451, 1452}


def initialize_mysql_database():
    if mysql_connector is None:
        raise RuntimeError('Install mysql-connector-python from requirements.txt to use MySQL.')
    if not re.fullmatch(r'[A-Za-z0-9_]+', MYSQL_DATABASE):
        raise ValueError('MYSQL_DATABASE may contain only letters, numbers, and underscores.')

    password = os.getenv('MYSQL_PASSWORD')
    if not password:
        raise RuntimeError('Set MYSQL_PASSWORD in the environment before starting the app with MySQL.')

    server_connection = mysql_connector.connect(
        host=os.getenv('MYSQL_HOST', '127.0.0.1'),
        port=int(os.getenv('MYSQL_PORT', '3306')),
        user=os.getenv('MYSQL_USER', 'root'),
        password=password,
    )
    try:
        cursor = server_connection.cursor()
        cursor.execute(
            f'CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4'
        )
    finally:
        server_connection.close()

    conn = get_db_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(320) NOT NULL UNIQUE,
                password CHAR(64) NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                user_email VARCHAR(320) NOT NULL,
                vehicle_name VARCHAR(255) NOT NULL,
                service_type VARCHAR(100) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'Pending',
                scheduled_for DATETIME NULL,
                notes TEXT NOT NULL,
                priority VARCHAR(20) NOT NULL DEFAULT 'Standard',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                active_scheduled_for DATETIME GENERATED ALWAYS AS (
                    CASE WHEN status = 'Cancelled' THEN NULL ELSE scheduled_for END
                ) STORED,
                UNIQUE KEY unique_active_booking_slot (active_scheduled_for),
                INDEX idx_bookings_user_email (user_email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')

        booking_columns = {row[0] for row in conn.execute('SHOW COLUMNS FROM bookings').fetchall()}
        for column, definition in (
            ('scheduled_for', 'DATETIME NULL'),
            ('notes', "TEXT NOT NULL DEFAULT ''"),
            ('priority', "VARCHAR(20) NOT NULL DEFAULT 'Standard'"),
        ):
            if column not in booking_columns:
                conn.execute(f'ALTER TABLE bookings ADD COLUMN {column} {definition}')
        if 'active_scheduled_for' not in booking_columns:
            conn.execute('''
                ALTER TABLE bookings ADD COLUMN active_scheduled_for DATETIME GENERATED ALWAYS AS (
                    CASE WHEN status = 'Cancelled' THEN NULL ELSE scheduled_for END
                ) STORED
            ''')
        booking_indexes = {row[2] for row in conn.execute('SHOW INDEX FROM bookings').fetchall()}
        if 'unique_active_booking_slot' not in booking_indexes:
            conn.execute('ALTER TABLE bookings ADD UNIQUE INDEX unique_active_booking_slot (active_scheduled_for)')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS app_migrations (
                migration_key VARCHAR(100) PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        migrated = conn.execute(
            'SELECT 1 FROM app_migrations WHERE migration_key=?',
            ('sqlite_to_mysql_v1',)
        ).fetchone()
        if not migrated:
            migrate_sqlite_data(conn)
            conn.execute(
                'INSERT IGNORE INTO app_migrations (migration_key) VALUES (?)',
                ('sqlite_to_mysql_v1',)
            )

        admin_email = 'admin@vehiclecare.com'
        admin_exists = conn.execute('SELECT 1 FROM users WHERE email=?', (admin_email,)).fetchone()
        if not admin_exists:
            conn.execute(
                'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
                ('Admin', admin_email, hash_password('admin123'), 'admin')
            )
        conn.commit()
    finally:
        conn.close()


def migrate_sqlite_data(mysql_connection):
    if not os.path.exists(DB_PATH):
        return

    sqlite_connection = sqlite3.connect(DB_PATH)
    try:
        tables = {
            row[0] for row in sqlite_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if 'users' in tables:
            for row in sqlite_connection.execute(
                'SELECT id, name, email, password, role FROM users'
            ).fetchall():
                mysql_connection.execute(
                    '''
                    INSERT IGNORE INTO users (id, name, email, password, role)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    row,
                )

        if 'bookings' in tables:
            columns = {
                row[1] for row in sqlite_connection.execute('PRAGMA table_info(bookings)').fetchall()
            }
            selected_columns = ['id', 'user_email', 'vehicle_name', 'service_type', 'status']
            fallback_values = {'scheduled_for': 'NULL', 'notes': "''", 'priority': "'Standard'"}
            selected_columns.extend(
                column if column in columns else f'{fallback_values[column]} AS {column}'
                for column in fallback_values
            )
            selected_columns.append('created_at')
            rows = sqlite_connection.execute(
                f"SELECT {', '.join(selected_columns)} FROM bookings"
            ).fetchall()
            for row in rows:
                mysql_connection.execute(
                    '''
                    INSERT IGNORE INTO bookings
                        (id, user_email, vehicle_name, service_type, status, scheduled_for, notes, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    row,
                )
    finally:
        sqlite_connection.close()


def init_db():
    if DB_BACKEND == 'mysql':
        initialize_mysql_database()
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            vehicle_name TEXT NOT NULL,
            service_type TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            scheduled_for TEXT,
            notes TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'Standard',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    booking_columns = {row[1] for row in conn.execute('PRAGMA table_info(bookings)')}
    for column, definition in (
        ('scheduled_for', 'TEXT'),
        ('notes', "TEXT NOT NULL DEFAULT ''"),
        ('priority', "TEXT NOT NULL DEFAULT 'Standard'"),
    ):
        if column not in booking_columns:
            conn.execute(f'ALTER TABLE bookings ADD COLUMN {column} {definition}')
    conn.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS unique_active_booking_slot
        ON bookings(scheduled_for)
        WHERE scheduled_for IS NOT NULL AND status != 'Cancelled'
    ''')
    conn.commit()

    admin_email = 'admin@vehiclecare.com'
    admin_exists = conn.execute('SELECT 1 FROM users WHERE email=?', (admin_email,)).fetchone()
    if not admin_exists:
        conn.execute(
            'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
            ('Admin', admin_email, hash_password('admin123'), 'admin')
        )
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(name, email, password):
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
            (name, email.lower(), hash_password(password), 'user')
        )
        conn.commit()
        return True
    except Exception as error:
        if is_integrity_error(error):
            return False
        raise
    finally:
        conn.close()


def authenticate_user(email, password):
    conn = get_db_connection()
    user = conn.execute(
        'SELECT name, email, role FROM users WHERE email=? AND password=?',
        (email.lower(), hash_password(password))
    ).fetchone()
    conn.close()
    if user:
        return {'name': user[0], 'email': user[1], 'role': user[2]}
    return None


def format_scheduled_for(booking_date, time_label):
    scheduled_time = datetime.strptime(time_label, '%I:%M %p').strftime('%H:%M')
    return f'{booking_date.isoformat()} {scheduled_time}'


def get_available_slots(booking_date, exclude_booking_id=None):
    conn = get_db_connection()
    query = '''
        SELECT scheduled_for FROM bookings
        WHERE scheduled_for LIKE ? AND status != 'Cancelled'
    '''
    params = [f'{booking_date.isoformat()}%']
    if exclude_booking_id is not None:
        query += ' AND id != ?'
        params.append(exclude_booking_id)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    occupied = {
        row[0].strftime('%Y-%m-%d %H:%M') if isinstance(row[0], datetime) else row[0]
        for row in rows
    }

    available = []
    for time_label in APPOINTMENT_TIMES:
        scheduled_for = format_scheduled_for(booking_date, time_label)
        slot_time = datetime.strptime(time_label, '%I:%M %p').time()
        if scheduled_for not in occupied and (booking_date != date.today() or slot_time > datetime.now().time()):
            available.append(time_label)
    return available


def save_booking(user_email, vehicle_name, service_type, scheduled_for, notes, priority):
    conn = get_db_connection()
    try:
        conn.execute('BEGIN IMMEDIATE')
        occupied = conn.execute(
            "SELECT 1 FROM bookings WHERE scheduled_for=? AND status != 'Cancelled'",
            (scheduled_for,)
        ).fetchone()
        if occupied:
            conn.rollback()
            return None

        cursor = conn.execute(
            '''
            INSERT INTO bookings
                (user_email, vehicle_name, service_type, status, scheduled_for, notes, priority)
            VALUES (?, ?, ?, 'Pending', ?, ?, ?)
            ''',
            (user_email, vehicle_name, service_type, scheduled_for, notes, priority)
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as error:
        conn.rollback()
        if is_integrity_error(error):
            return None
        raise
    finally:
        conn.close()


def get_user_bookings(user_email):
    conn = get_db_connection()
    rows = conn.execute(
        '''
        SELECT id, vehicle_name, service_type, scheduled_for, status, priority, notes, created_at
        FROM bookings WHERE user_email=? ORDER BY created_at DESC
        ''',
        (user_email,)
    ).fetchall()
    conn.close()
    return pd.DataFrame(
        rows,
        columns=['id', 'vehicle_name', 'service_type', 'scheduled_for', 'status', 'priority', 'notes', 'created_at']
    )


def cancel_booking(booking_id, user_email):
    conn = get_db_connection()
    cursor = conn.execute(
        "UPDATE bookings SET status='Cancelled' WHERE id=? AND user_email=? AND status IN ('Pending', 'Confirmed', 'Booked')",
        (booking_id, user_email)
    )
    conn.commit()
    changed = cursor.rowcount > 0
    conn.close()
    return changed


def reschedule_booking(booking_id, user_email, scheduled_for):
    conn = get_db_connection()
    try:
        conn.execute('BEGIN IMMEDIATE')
        booking = conn.execute(
            "SELECT status FROM bookings WHERE id=? AND user_email=?",
            (booking_id, user_email)
        ).fetchone()
        if not booking or booking[0] not in ('Pending', 'Confirmed', 'Booked'):
            conn.rollback()
            return False

        occupied = conn.execute(
            "SELECT 1 FROM bookings WHERE scheduled_for=? AND status != 'Cancelled' AND id != ?",
            (scheduled_for, booking_id)
        ).fetchone()
        if occupied:
            conn.rollback()
            return False

        conn.execute('UPDATE bookings SET scheduled_for=? WHERE id=?', (scheduled_for, booking_id))
        conn.commit()
        return True
    except Exception as error:
        conn.rollback()
        if is_integrity_error(error):
            return False
        raise
    finally:
        conn.close()


def update_booking_status(booking_id, status):
    if status not in BOOKING_STATUSES:
        return False
    conn = get_db_connection()
    try:
        conn.execute('BEGIN IMMEDIATE')
        cursor = conn.execute('UPDATE bookings SET status=? WHERE id=?', (status, booking_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as error:
        conn.rollback()
        if is_integrity_error(error):
            return False
        raise
    finally:
        conn.close()


def get_all_bookings():
    conn = get_db_connection()
    rows = conn.execute(
        '''
        SELECT id, user_email, vehicle_name, service_type, scheduled_for, status, priority, notes, created_at
        FROM bookings ORDER BY created_at DESC
        '''
    ).fetchall()
    conn.close()
    return pd.DataFrame(
        rows,
        columns=['id', 'user_email', 'vehicle_name', 'service_type', 'scheduled_for', 'status', 'priority', 'notes', 'created_at']
    )


def get_booking_summary():
    bookings = get_all_bookings()
    if bookings.empty:
        return {'total': 0, 'oil': 0, 'inspection': 0, 'wash': 0}
    return {
        'total': len(bookings),
        'oil': int((bookings['service_type'].str.contains('Oil', case=False, na=False)).sum()),
        'inspection': int((bookings['service_type'].str.contains('Inspection', case=False, na=False)).sum()),
        'wash': int((bookings['service_type'].str.contains('Wash', case=False, na=False)).sum()),
    }


def train_model_from_csv():
    df = pd.read_csv('engine_data.csv')
    df['Temperature_difference'] = df['Coolant temp'] - df['lub oil temp']
    X = df[FEATURE_COLUMNS]
    y = df['Engine Condition']

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    return model


def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as file:
                    return pickle.load(file)
            except Exception:
                continue
    return train_model_from_csv()


init_db()
model = load_model()


def get_recommendation(result, confidence):
    if result == 0:
        return {
            'title': 'Healthy Vehicle Status',
            'message': 'Your vehicle is currently operating within a stable condition. Continue monitoring and keep the regular maintenance schedule.',
            'box_class': 'recommendation-box',
        }
    if confidence >= 0.75:
        return {
            'title': 'High Risk Detected',
            'message': 'The model identifies a strong likelihood of mechanical issues. Schedule a service inspection as soon as possible.',
            'box_class': 'danger-box',
        }
    return {
        'title': 'Moderate Risk Detected',
        'message': 'A potential issue has been identified. Consider an early inspection and monitor the vehicle closely before the next trip.',
        'box_class': 'warning-box',
    }


def answer_chatbot(user_message, risk_status):
    msg = user_message.lower()
    if 'oil' in msg:
        service = 'Oil Change'
    elif 'wash' in msg or 'clean' in msg:
        service = 'Car Wash'
    elif 'inspection' in msg or 'check' in msg:
        service = 'Vehicle Inspection'
    else:
        service = 'General Service'

    if 'book' in msg or 'schedule' in msg or 'appointment' in msg or 'service' in msg:
        if risk_status == 'At Risk':
            return f"I recommend booking a {service} appointment soon. Your vehicle health suggests a maintenance review is necessary before the next trip."
        return f"I can help you schedule a {service} appointment. Please confirm the vehicle and preferred date."

    if 'status' in msg or 'health' in msg:
        return f"Your current vehicle status is {risk_status}. The predictive model is monitoring engine performance and maintenance risk."

    return f"I can assist with vehicle health checks, maintenance booking, and service recommendations. For example: 'Book oil change' or 'Check status'."


st.set_page_config(page_title='AI Vehicle Predictive Maintenance', page_icon='🚗', layout='wide')

st.markdown(
    '''
    <style>
    .main { background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        border-radius: 18px;
        padding: 1.8rem 1.5rem;
        color: white;
        box-shadow: 0 15px 35px rgba(29, 78, 216, 0.15);
    }
    .hero-box h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
    .hero-box p { margin-top: 0.5rem; color: rgba(255,255,255,0.9); }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.1rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .metric-label {
        color: #475569; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0f172a; margin-top: 0.3rem; }
    .recommendation-box { background: #f0fdf4; border-left: 5px solid #22c55e; padding: 1rem 1.1rem; border-radius: 12px; color: #14532d; }
    .warning-box { background: #fff7ed; border-left: 5px solid #f97316; padding: 1rem 1.1rem; border-radius: 12px; color: #9a4d00; }
    .danger-box { background: #fef2f2; border-left: 5px solid #ef4444; padding: 1rem 1.1rem; border-radius: 12px; color: #7f1d1d; }
    </style>
    ''',
    unsafe_allow_html=True,
)


if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'user_email' not in st.session_state:
    st.session_state.user_email = ''
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'user'
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False
if 'admin_name' not in st.session_state:
    st.session_state.admin_name = ''
if 'vehicle_name' not in st.session_state:
    st.session_state.vehicle_name = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = 0.0


def main():
    st.markdown(
        '''
        <div class="hero-box">
            <h1>AI Vehicle Predictive Maintenance</h1>
            <p>Machine learning-powered vehicle health monitoring, risk detection, and smart service booking.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.sidebar.title('Vehicle AI Assistant')
    st.sidebar.caption('Final-year predictive maintenance prototype')

    if not st.session_state.user_name:
        auth_mode = st.sidebar.radio('Authentication', ['Login', 'Sign Up'])
        if auth_mode == 'Sign Up':
            with st.sidebar.form('signup_form'):
                name = st.text_input('Full name')
                email = st.text_input('Email')
                password = st.text_input('Password', type='password')
                submit = st.form_submit_button('Create account')
                if submit:
                    if name and email and password:
                        if register_user(name, email, password):
                            user = authenticate_user(email, password)
                            if user:
                                st.session_state.user_name = user['name']
                                st.session_state.user_email = user['email']
                                st.session_state.user_role = user['role']
                                st.session_state.vehicle_name = 'Vehicle'
                                st.rerun()
                        else:
                            st.sidebar.error('This email is already registered.')
        else:
            with st.sidebar.form('signin_form'):
                email = st.text_input('Email')
                password = st.text_input('Password', type='password')
                submit = st.form_submit_button('Login')
                if submit:
                    user = authenticate_user(email, password)
                    if user and user['role'] == 'user':
                        st.session_state.user_name = user['name']
                        st.session_state.user_email = user['email']
                        st.session_state.user_role = user['role']
                        st.session_state.vehicle_name = 'Vehicle'
                        st.rerun()
                    elif user:
                        st.sidebar.error('Admins must sign in from the Admin tab.')
                    else:
                        st.sidebar.error('Invalid email or password.')
    else:
        st.sidebar.success(f'Logged in: {st.session_state.user_name}')
        st.sidebar.info(f'Role: {st.session_state.user_role}')
        st.sidebar.info(f'Vehicle: {st.session_state.vehicle_name}')
        if st.sidebar.button('Logout'):
            st.session_state.user_name = ''
            st.session_state.user_email = ''
            st.session_state.user_role = 'user'
            st.session_state.vehicle_name = ''
            st.rerun()

    st.sidebar.markdown('### Feature Descriptions')
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f'**{feature}:** {description}')

    tabs = st.tabs(['Overview', 'Predictive Analysis', 'AI Booking', 'Admin'])

    with tabs[0]:
        st.markdown('### Vehicle health overview')
        st.write('Use Predictive Analysis to check engine condition and AI Booking to request a maintenance service.')

    with tabs[1]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('### Enter vehicle parameters')
            engine_rpm = st.slider('Engine RPM', min_value=float(custom_ranges['Engine rpm'][0]), max_value=float(custom_ranges['Engine rpm'][1]), value=float(custom_ranges['Engine rpm'][1] / 2), step=1.0)
            lub_oil_pressure = st.slider('Lub Oil Pressure', min_value=custom_ranges['Lub oil pressure'][0], max_value=custom_ranges['Lub oil pressure'][1], value=(custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2, step=0.01)
            fuel_pressure = st.slider('Fuel Pressure', min_value=custom_ranges['Fuel pressure'][0], max_value=custom_ranges['Fuel pressure'][1], value=(custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2, step=0.01)
            coolant_pressure = st.slider('Coolant Pressure', min_value=custom_ranges['Coolant pressure'][0], max_value=custom_ranges['Coolant pressure'][1], value=(custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2, step=0.01)
            lub_oil_temp = st.slider('Lub Oil Temperature', min_value=custom_ranges['lub oil temp'][0], max_value=custom_ranges['lub oil temp'][1], value=(custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2, step=0.1)
            coolant_temp = st.slider('Coolant Temperature', min_value=custom_ranges['Coolant temp'][0], max_value=custom_ranges['Coolant temp'][1], value=(custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2, step=0.1)
            temp_difference = coolant_temp - lub_oil_temp
            st.caption('Temperature difference is calculated automatically from coolant and oil temperatures.')

        with col2:
            st.markdown('### Model status')
            st.markdown("<div class='metric-card'><div class='metric-label'>AI Model</div><div class='metric-value'>Gradient Boosting</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-card' style='margin-top: 1rem;'><div class='metric-label'>Monitoring</div><div class='metric-value'>Live Risk Scan</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-card' style='margin-top: 1rem;'><div class='metric-label'>Service Focus</div><div class='metric-value'>Inspection / Oil Care</div></div>", unsafe_allow_html=True)

        if st.button('Predict Engine Condition', use_container_width=True):
            result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
            recommendation = get_recommendation(result, confidence)
            st.session_state.last_prediction = result
            st.session_state.last_confidence = confidence
            st.session_state.last_recommendation = recommendation

        if st.session_state.last_prediction is not None:
            result = st.session_state.last_prediction
            confidence = st.session_state.last_confidence
            recommendation = st.session_state.last_recommendation

            st.markdown('### Prediction Result')
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                status_label = 'Healthy' if result == 0 else 'At Risk'
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Vehicle Status</div><div class='metric-value'>{status_label}</div></div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Prediction Confidence</div><div class='metric-value'>{confidence:.0%}</div></div>", unsafe_allow_html=True)
            with col_c:
                action_label = 'Monitor' if result == 0 else 'Inspection'
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Recommended Action</div><div class='metric-value'>{action_label}</div></div>", unsafe_allow_html=True)

            st.markdown(f"<div class='{recommendation['box_class']}'><strong>{recommendation['title']}</strong><br>{recommendation['message']}</div>", unsafe_allow_html=True)

            if result == 0:
                st.success(f'The model predicts a normal engine condition with {confidence:.0%} confidence.')
            else:
                st.warning(f'Warning! A maintenance issue may be developing. The model confidence is {confidence:.0%}.')

    with tabs[2]:
        st.markdown('### AI Service Assistant')
        if st.session_state.user_name:
            st.write(f'Welcome, {st.session_state.user_name}. I can help you schedule a service for {st.session_state.vehicle_name}.')
        else:
            st.write('Sign in from the sidebar to request an appointment and manage your bookings.')

        if st.session_state.last_prediction == 1:
            st.warning('Your latest vehicle check is at risk. Vehicle Inspection is recommended and your request will be marked high priority.')
        elif st.session_state.last_prediction == 0:
            st.info('Your latest vehicle check is healthy. You can still request routine maintenance.')

        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div style="text-align:right;"><div style="display:inline-block;background:#dbeafe;padding:10px 14px;border-radius:12px;max-width:70%;">{msg["text"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:left;"><div style="display:inline-block;background:#f1f5f9;padding:10px 14px;border-radius:12px;max-width:80%;">{msg["text"]}</div></div>', unsafe_allow_html=True)

        user_input = st.chat_input('Ask the AI assistant about maintenance or booking')
        if user_input:
            risk_status = 'At Risk' if st.session_state.last_prediction == 1 else 'Healthy'
            response = answer_chatbot(user_input, risk_status)
            st.session_state.chat_history.append({'role': 'user', 'text': user_input})
            st.session_state.chat_history.append({'role': 'assistant', 'text': response})
            st.rerun()

        st.markdown('### Request an appointment')
        service_options = ['Oil Change', 'Vehicle Inspection', 'Car Wash', 'General Service']
        recommended_service = 'Vehicle Inspection' if st.session_state.last_prediction == 1 else 'Oil Change'
        booking_service = st.selectbox(
            'Service',
            service_options,
            index=service_options.index(recommended_service),
            key='booking_service',
        )
        booking_vehicle = st.text_input(
            'Vehicle name or model',
            value=st.session_state.vehicle_name or 'Vehicle',
            key='booking_vehicle',
        )
        booking_date = st.date_input(
            'Preferred date',
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            key='booking_date',
        )
        available_slots = get_available_slots(booking_date)
        slot_options = available_slots or ['No times available']
        selected_time = st.selectbox(
            'Available appointment time',
            slot_options,
            disabled=not available_slots,
            key='booking_time',
        )
        booking_notes = st.text_area('Notes for the service team', key='booking_notes')

        if not available_slots:
            st.info('There are no appointment times available on this date. Choose another date.')

        if st.button('Request appointment', type='primary', disabled=not available_slots):
            if st.session_state.user_name:
                if not booking_vehicle.strip():
                    st.error('Enter a vehicle name or model to continue.')
                else:
                    scheduled_for = format_scheduled_for(booking_date, selected_time)
                    priority = 'High' if st.session_state.last_prediction == 1 else 'Standard'
                    booking_id = save_booking(
                        st.session_state.user_email,
                        booking_vehicle.strip(),
                        booking_service,
                        scheduled_for,
                        booking_notes.strip(),
                        priority,
                    )
                    if booking_id is None:
                        st.warning('That appointment time was just taken. Choose another available time.')
                    else:
                        st.session_state.booking_confirmation = {
                            'id': booking_id,
                            'user_email': st.session_state.user_email,
                            'service': booking_service,
                            'vehicle': booking_vehicle.strip(),
                            'scheduled_for': scheduled_for,
                            'priority': priority,
                        }
                        st.rerun()
            else:
                st.warning('Please login in the sidebar before booking a service.')

        confirmation = st.session_state.get('booking_confirmation')
        if confirmation and confirmation['user_email'] == st.session_state.user_email:
            st.success(
                f"Request #{confirmation['id']} received: {confirmation['service']} for "
                f"{confirmation['vehicle']} on {confirmation['scheduled_for']}. "
                f"Priority: {confirmation['priority']}. Awaiting admin confirmation."
            )

        st.markdown('### Your bookings')
        if st.session_state.user_email:
            user_bookings = get_user_bookings(st.session_state.user_email)
            if user_bookings.empty:
                st.info('Your appointment history will appear here.')
            else:
                st.dataframe(user_bookings, use_container_width=True, hide_index=True)
                manageable_bookings = user_bookings[
                    user_bookings['status'].isin(['Pending', 'Confirmed', 'Booked'])
                ]
                if not manageable_bookings.empty:
                    booking_details = manageable_bookings.set_index('id').to_dict(orient='index')
                    booking_ids = manageable_bookings['id'].tolist()
                    selected_booking_id = st.selectbox(
                        'Booking to manage',
                        booking_ids,
                        format_func=lambda booking_id: (
                            f"#{booking_id} | {booking_details[booking_id]['service_type']} | "
                            f"{booking_details[booking_id]['scheduled_for'] or 'Time not selected'}"
                        ),
                        key='manage_booking_id',
                    )
                    selected_booking = booking_details[selected_booking_id]
                    if selected_booking['scheduled_for']:
                        reschedule_date = st.date_input(
                            'New appointment date',
                            value=date.today() + timedelta(days=1),
                            min_value=date.today(),
                            key='reschedule_date',
                        )
                        reschedule_slots = get_available_slots(reschedule_date, selected_booking_id)
                        reschedule_options = reschedule_slots or ['No times available']
                        reschedule_time = st.selectbox(
                            'New appointment time',
                            reschedule_options,
                            disabled=not reschedule_slots,
                            key='reschedule_time',
                        )
                        if st.button('Reschedule booking', disabled=not reschedule_slots):
                            new_scheduled_for = format_scheduled_for(reschedule_date, reschedule_time)
                            if reschedule_booking(
                                selected_booking_id,
                                st.session_state.user_email,
                                new_scheduled_for,
                            ):
                                st.success(f'Booking #{selected_booking_id} rescheduled to {new_scheduled_for}.')
                                st.rerun()
                            else:
                                st.warning('That time is no longer available. Choose another appointment time.')
                    else:
                        st.info('This older booking has no appointment time. You can cancel it below.')

                    if st.button('Cancel selected booking'):
                        if cancel_booking(selected_booking_id, st.session_state.user_email):
                            st.success(f'Booking #{selected_booking_id} cancelled. Its appointment time is available again.')
                            st.rerun()
                        else:
                            st.warning('This booking can no longer be cancelled.')
        else:
            st.info('Sign in to view, cancel, or reschedule your bookings.')

    with tabs[3]:
        if not st.session_state.admin_authenticated:
            st.markdown('### Admin sign in')
            with st.form('admin_signin_form'):
                email = st.text_input('Admin email')
                password = st.text_input('Admin password', type='password')
                submit = st.form_submit_button('Sign in as admin')
                if submit:
                    admin = authenticate_user(email, password)
                    if admin and admin['role'] == 'admin':
                        st.session_state.admin_authenticated = True
                        st.session_state.admin_name = admin['name']
                        st.rerun()
                    st.error('Invalid admin email or password.')
        else:
            st.markdown('### Admin Dashboard')
            st.write(f"Signed in as {st.session_state.admin_name}.")
            if st.button('Admin logout'):
                st.session_state.admin_authenticated = False
                st.session_state.admin_name = ''
                st.rerun()

            summary = get_booking_summary()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Bookings</div><div class='metric-value'>{summary['total']}</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Oil Services</div><div class='metric-value'>{summary['oil']}</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Inspections</div><div class='metric-value'>{summary['inspection']}</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Car Wash</div><div class='metric-value'>{summary['wash']}</div></div>", unsafe_allow_html=True)

            bookings = get_all_bookings()
            if bookings.empty:
                st.info('No service records yet.')
            else:
                st.markdown('### Booking queue')
                st.dataframe(bookings, use_container_width=True, hide_index=True)

                booking_details = bookings.set_index('id').to_dict(orient='index')
                booking_ids = bookings['id'].tolist()
                selected_booking_id = st.selectbox(
                    'Booking to update',
                    booking_ids,
                    format_func=lambda booking_id: (
                        f"#{booking_id} | {booking_details[booking_id]['user_email']} | "
                        f"{booking_details[booking_id]['service_type']} | "
                        f"{booking_details[booking_id]['scheduled_for'] or 'No time selected'}"
                    ),
                    key='admin_booking_id',
                )
                current_status = booking_details[selected_booking_id]['status']
                status_index = BOOKING_STATUSES.index(current_status) if current_status in BOOKING_STATUSES else 0
                with st.form('admin_booking_status_form'):
                    updated_status = st.selectbox(
                        'Booking status',
                        BOOKING_STATUSES,
                        index=status_index,
                    )
                    update_status = st.form_submit_button('Update booking status')
                    if update_status:
                        if update_booking_status(selected_booking_id, updated_status):
                            st.success(f'Booking #{selected_booking_id} updated to {updated_status}.')
                            st.rerun()
                        st.error('Could not update this booking. Its time may have been taken by another appointment.')

                service_counts = bookings['service_type'].value_counts()
                st.bar_chart(service_counts)


# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_df = pd.DataFrame([
        {
            'Engine rpm': float(engine_rpm),
            'Lub oil pressure': float(lub_oil_pressure),
            'Fuel pressure': float(fuel_pressure),
            'Coolant pressure': float(coolant_pressure),
            'lub oil temp': float(lub_oil_temp),
            'Coolant temp': float(coolant_temp),
            'Temperature_difference': float(temp_difference),
        }
    ])
    prediction = model.predict(input_df[FEATURE_COLUMNS])
    confidence = model.predict_proba(input_df[FEATURE_COLUMNS])[:, 1]
    return prediction[0], confidence[0]


if __name__ == '__main__':
    main()
