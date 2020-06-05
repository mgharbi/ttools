"""Utilities for sqlite database interaction."""
import os
import sqlalchemy as db
import pandas as pd


class SQLiteDatabase(object):
    def __init__(self, fname):
        fname = os.path.abspath(fname)
        dirname = os.path.dirname(fname)
        os.makedirs(dirname, exist_ok=True)
        ext = os.path.splitext(fname)[-1]
        if ext == "":
            fname += ".sqlite"
        else:
            if ext != ".sqlite":
                msg = "Expected .sqlite extension for database"
                raise ValueError(msg)
        self.fname = fname

        self.engine = db.create_engine('sqlite:///' + fname)
        self.db_conn = self.engine.connect()

    def __del__(self):
        self.db_conn.close()

    def append_row(self, data, table_name):
        """Appends a row of data to a database table.
        Args:
            data(dict): data to insert.
            table_name(str): name of the table to update.
        """

        data = pd.DataFrame([data])
        data.to_sql(
            table_name, self.db_conn, if_exists="append", index=False)

    def read_table(self, table_name):
        try:
            return pd.read_sql_table(table_name, self.db_conn)
        except ValueError:  # no table found
            return None


