from finkvra.alonstream.alpipeline import ALPipeline
import importlib.resources as pkg_resources
import finkvra.data

def configfile():
    return pkg_resources.files(finkvra.data).joinpath('test_config.yaml')

def test_alpipeline_initialization():
    """Test ALPipeline initialization."""
    pipeline = ALPipeline(configfile=configfile(),)
    assert isinstance(pipeline, ALPipeline)