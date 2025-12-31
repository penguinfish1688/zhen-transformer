from transformer.data.dataset import download_wmt_sample

def test_download_wmt_sample():
    """Test downloading WMT sample dataset"""
    chinese_sentences, english_sentences = download_wmt_sample()
    print(chinese_sentences[:3], english_sentences[:3])

test_download_wmt_sample()
