"""Test the WMT download functionality"""
from transformer.data.dataset import download_wmt_sample, download_sample_data

def test_download_formats():
    """Verify both functions return the same format"""
    
    print("\n" + "="*70)
    print("Testing format compatibility...")
    print("="*70)
    
    # Test sample data
    print("\n1. Testing sample data format...")
    zh_sample, en_sample = download_sample_data()
    print(f"   Type: {type(zh_sample)}, {type(en_sample)}")
    print(f"   Length: {len(zh_sample)}, {len(en_sample)}")
    print(f"   First item type: {type(zh_sample[0])}, {type(en_sample[0])}")
    print(f"   Sample: {zh_sample[0]} -> {en_sample[0]}")
    
    # Test WMT download
    print("\n2. Testing WMT download format...")
    zh_wmt, en_wmt = download_wmt_sample(num_samples=100)
    print(f"   Type: {type(zh_wmt)}, {type(en_wmt)}")
    print(f"   Length: {len(zh_wmt)}, {len(en_wmt)}")
    print(f"   First item type: {type(zh_wmt[0])}, {type(en_wmt[0])}")
    print(f"   Sample: {zh_wmt[0]} -> {en_wmt[0]}")
    
    # Verify format compatibility
    print("\n3. Verifying format compatibility...")
    assert isinstance(zh_sample, list), "Sample Chinese should be list"
    assert isinstance(en_sample, list), "Sample English should be list"
    assert isinstance(zh_wmt, list), "WMT Chinese should be list"
    assert isinstance(en_wmt, list), "WMT English should be list"
    
    assert len(zh_sample) == len(en_sample), "Sample pairs should match"
    assert len(zh_wmt) == len(en_wmt), "WMT pairs should match"
    
    assert all(isinstance(s, str) for s in zh_sample), "All Chinese samples should be strings"
    assert all(isinstance(s, str) for s in en_sample), "All English samples should be strings"
    assert all(isinstance(s, str) for s in zh_wmt), "All Chinese WMT should be strings"
    assert all(isinstance(s, str) for s in en_wmt), "All English WMT should be strings"
    
    print("   ✅ Format check passed!")
    print("   ✅ Both functions return: Tuple[List[str], List[str]]")
    
    # Show some examples
    print("\n4. Sample WMT sentences:")
    for i in range(min(5, len(zh_wmt))):
        print(f"   [{i+1}] ZH: {zh_wmt[i]}")
        print(f"       EN: {en_wmt[i]}")
    
    print("\n" + "="*70)
    print("✅ ALL FORMAT TESTS PASSED!")
    print("="*70)

if __name__ == "__main__":
    test_download_formats()
