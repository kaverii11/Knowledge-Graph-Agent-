from neo4j import GraphDatabase


class Neo4jConnection:
    """Thin wrapper around the Neo4j driver, shared by ingestion, the CLI and the Streamlit app."""

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__password = pwd
        self.__driver = None
        self._connect()

    def _connect(self):
        self.__driver = GraphDatabase.driver(
            self.__uri, auth=(self.__user, self.__password), max_connection_lifetime=200
        )
        self.__driver.verify_connectivity()

    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            self.__driver = None

    def _run(self, query, parameters, read_only):
        with self.__driver.session() as session:
            work = lambda tx: [record.data() for record in tx.run(query, parameters or {})]
            if read_only:
                return session.execute_read(work)
            return session.execute_write(work)

    def execute_query(self, query, parameters=None, read_only=False):
        """Runs a query, reconnecting once if the connection went stale (e.g. Aura idle timeout)."""
        if self.__driver is None:
            self._connect()
        try:
            return self._run(query, parameters, read_only)
        except Exception:
            self.close()
            self._connect()
            return self._run(query, parameters, read_only)
