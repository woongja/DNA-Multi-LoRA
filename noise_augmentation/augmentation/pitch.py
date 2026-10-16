from .base import BaseAugmentor
from .utils import librosa_to_pydub
import random
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from audiomentations import PitchShift
import logging
logger = logging.getLogger(__name__)

class PitchAugmentor(BaseAugmentor):
    """
    Pitch augmentation
    Generates versions of audio pitch-shifted by +-2 to +-5 semitones
    """
    def __init__(self, config: dict):
        """
        This method initializes the `PitchAugmentor` object.
        min_semitones: float • unit: semitones • range: [-24.0, 24.0]
        max_semitones: float • unit: semitones • range: [-24.0, 24.0]
        :param config: dict, configuration dictionary
        """
        super().__init__(config)
        # ICASSP 2026 Table 1: shift by +-2 to +-5 semitones (|n_steps| in [2, 5], random sign)
        self.min_abs_semitones = config.get("min_abs_semitones", 2)
        self.max_abs_semitones = config.get("max_abs_semitones", 5)
        
        # self.pitch_augmentor = PitchShift(
        #     min_semitones=self.min_semitones,
        #     max_semitones=self.max_semitones,
        #     method="librosa_phase_vocoder",
        #     p=1.0
        # )

    def transform(self):
        """
        Transform the audio by pitch shifting
        
        Applies a random shift with |n_steps| in [min_abs_semitones, max_abs_semitones]
        :return: The pitch-shifted audio segment
        """
        n_steps = np.random.uniform(self.min_abs_semitones, self.max_abs_semitones)
        n_steps *= np.random.choice([-1, 1])
        # librosa로 적용
        augmented_data = librosa.effects.pitch_shift(y = self.data, sr = self.sr, n_steps=n_steps)
        # Transform to pydub audio segment
        self.augmented_audio = librosa_to_pydub(augmented_data, sr=self.sr)
        self.ratio = f"pitch:{n_steps}"
 