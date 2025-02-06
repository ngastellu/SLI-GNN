import sqlite3

def build_db(db_name):
    # Step 1: Connect to a new (or existing) SQLite database
    conn = sqlite3.connect(db_name + ".db")
    cursor = conn.cursor()

    # Step 2: Read and execute the .sql file
    with open(f"{db_name}.sql", "r", encoding="utf-8") as sql_file:
        sql_script = sql_file.read()

    cursor.executescript(sql_script)  # Executes all SQL commands in the file

    # Commit and close connection
    conn.commit()
    conn.close()

    print("Database successfully created and populated!")



def print_db_tables_headers(db_name):
    # Connect to the database that was just created
    conn = sqlite3.connect(db_name + ".db")
    cursor = conn.cursor()

    # List all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    print("Tables:", table_names)

    # Get columns for each table
    for table in table_names:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Table: {table}, Columns: {columns}")

    # Close connection
    conn.close()

sql_scripts = ['cepdb_2012-10-24.sql', 'cepdb_2013-06-21.sql']

for s in sql_scripts:
    print(f'******************* {s} *******************')
    db_name = s.split('.')[0]
    build_db(db_name)
    print_db_tables_headers(db_name)