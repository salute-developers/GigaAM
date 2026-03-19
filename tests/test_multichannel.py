import logging
import os
import tempfile
import urllib.request
from typing import List, Tuple

import numpy as np
import pytest
import soundfile as sf
import torch

import gigaam


def download_long_audio():
    """Download test audio file if not exists"""
    audio_file = "long_example.wav"
    if not os.path.exists(audio_file):
        url = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/long_example.wav"
        urllib.request.urlretrieve(url, audio_file)
    assert os.path.exists(audio_file), "Long audio file not found"
    return audio_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stereo_from_mono(audio: np.ndarray, offset_seconds: float = 0.0, sr: int = 16000) -> np.ndarray:
    """
    Create stereo audio from mono by duplicating with optional time offset.
    Channel 0: original audio
    Channel 1: audio shifted by offset_seconds
    """
    offset_samples = int(offset_seconds * sr)
    
    # Create two channels
    channel_0 = audio.copy()
    
    # Channel 1: shift audio by offset
    if offset_samples > 0:
        # Pad beginning with zeros
        channel_1 = np.pad(audio, (offset_samples, 0), mode='constant')
        # Trim to same length
        channel_1 = channel_1[:len(audio)]
    elif offset_samples < 0:
        # Shift left (remove from beginning)
        channel_1 = audio[-offset_samples:]
        # Pad end with zeros
        channel_1 = np.pad(channel_1, (0, -offset_samples), mode='constant')
    else:
        channel_1 = audio.copy()
    
    # Ensure same length
    min_len = min(len(channel_0), len(channel_1))
    channel_0 = channel_0[:min_len]
    channel_1 = channel_1[:min_len]
    
    # Stack into stereo: (2, samples)
    stereo = np.stack([channel_0, channel_1], axis=0)
    return stereo


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt", "v3_ctc"])
def test_transcribe_multichannel_stereo(revision):
    """Test multichannel transcription with stereo file created from mono"""
    # Download test audio
    mono_file = download_long_audio()
    
    # Load mono audio
    audio, sr = sf.read(mono_file)
    
    # Create stereo file with offset (channel 1 starts 5 seconds later)
    stereo_audio = create_stereo_from_mono(audio, offset_seconds=5.0, sr=sr)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file = f.name
    try:
        sf.write(temp_file, stereo_audio.T, sr)  # soundfile expects (samples, channels)
        
        # Load model
        model = gigaam.load_model(revision)
        
        # Test with stereo file
        results = model.transcribe_multichannel(temp_file, batch_size=4)
        
        assert isinstance(results, list), "Should return list of segments"
        assert len(results) > 0, "Should have at least one segment"
        
        # Check structure
        for seg in results:
            assert "channel" in seg, "Missing channel key"
            assert "transcription" in seg, "Missing transcription key"
            assert "boundaries" in seg, "Missing boundaries key"
            assert seg["channel"] in [0, 1], f"Invalid channel: {seg['channel']}"
            start, end = seg["boundaries"]
            assert start < end, f"Invalid boundaries: {start} >= {end}"
        
        logger.info(f"Multichannel test: {len(results)} segments for stereo file")
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@pytest.mark.parametrize("revision", ["v3_e2e_rnnt"])
def test_transcribe_multichannel_list(revision):
    """Test multichannel transcription with list of separate files"""
    # Download test audio
    mono_file = download_long_audio()
    
    # Load mono audio
    audio, sr = sf.read(mono_file)
    
    # Split into two parts with offset
    split_point = len(audio) // 2
    audio_0 = audio[:split_point]
    audio_1 = audio[split_point:]
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f0, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
        temp_file0 = f0.name
        temp_file1 = f1.name
    try:
        sf.write(temp_file0, audio_0, sr)
        sf.write(temp_file1, audio_1, sr)
        
        # Load model
        model = gigaam.load_model(revision)
        
        # Test with list of files
        results = model.transcribe_multichannel([temp_file0, temp_file1], batch_size=4)
        
        assert isinstance(results, list), "Should return list of segments"
        assert len(results) > 0, "Should have at least one segment"
        
        # Check structure
        for seg in results:
            assert "channel" in seg, "Missing channel key"
            assert "transcription" in seg, "Missing transcription key"
            assert "boundaries" in seg, "Missing boundaries key"
            assert seg["channel"] in [0, 1], f"Invalid channel: {seg['channel']}"
        
        logger.info(f"Multichannel test: {len(results)} segments for list of files")
        
    finally:
        for fname in [temp_file0, temp_file1]:
            if os.path.exists(fname):
                os.remove(fname)


def test_multichannel_channel_ordering():
    """Test that channels are correctly identified and ordered"""
    # Download test audio
    mono_file = download_long_audio()
    
    # Load mono audio
    audio, sr = sf.read(mono_file)
    
    # Create stereo: channel 0 = first half, channel 1 = second half
    split_point = len(audio) // 2
    channel_0 = audio[:split_point]
    channel_1 = audio[split_point:]
    
    # Pad to same length
    max_len = max(len(channel_0), len(channel_1))
    channel_0 = np.pad(channel_0, (0, max_len - len(channel_0)), mode='constant')
    channel_1 = np.pad(channel_1, (0, max_len - len(channel_1)), mode='constant')
    
    stereo_audio = np.stack([channel_0, channel_1], axis=0)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file = f.name
    try:
        sf.write(temp_file, stereo_audio.T, sr)
        
        model = gigaam.load_model("v3_e2e_rnnt")
        results = model.transcribe_multichannel(temp_file, batch_size=4)
        
        # Check that we have segments from both channels
        channels_found = set(seg["channel"] for seg in results)
        assert 0 in channels_found, "Should have segments from channel 0"
        assert 1 in channels_found, "Should have segments from channel 1"
        
        # Check ordering: segments should be sorted by time
        for i in range(len(results) - 1):
            assert results[i]["boundaries"][0] <= results[i+1]["boundaries"][0], \
                "Segments should be sorted by start time"
        
        logger.info(f"Channel ordering test: channels {channels_found}, {len(results)} segments")
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
