"""
连接池 - 为并发请求提供独立的 OpenAI client 实例
每个线程从池中取一个 client 使用，用完归还，避免多线程共享连接的问题
"""
import threading
from openai import OpenAI


class ConnectionPool:
    def __init__(self, api_key: str, base_url: str, pool_size: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.pool_size = pool_size
        self._pool: list[OpenAI] = []
        self._lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.pool_size):
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._pool.append(client)

    def get_connection(self) -> OpenAI:
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def return_connection(self, client: OpenAI):
        with self._lock:
            if len(self._pool) < self.pool_size:
                self._pool.append(client)
