import pytest

from expert.data.video_reader import Cache


@pytest.fixture()
def cache():
    cache = Cache(512)
    return cache


def test_capacity(cache):
    assert cache.capacity == 512


def test_size(cache):
    assert cache.size == 0


def test_put(cache):
    cache.put(0, [0])
    assert cache.size == 1


def test_get(cache):
    cache.put(0, [0])
    value = cache.get(0)
    assert value == [0]
