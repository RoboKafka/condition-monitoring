import unittest
import pandas as pd
from vib_analyzer import VibrationAnalyzer

class TestVibrationAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Time Stamp': ['13:34:14', '13:34:15', '13:34:16', '13:34:17', '13:34:18'],
            ' X-axis': [0.2274, 0.5596, 1.1617, 1.0181, 0.2385],
            ' Y-axis': [2.048, 2.048, 2.048, 2.048, 2.048],
            ' Z-axis': [0.8902, 1.0181, 1.0299, 0.6432, 0.9981]
        })
        self.vibration_analyzer = VibrationAnalyzer(df=self.df)
    
    def test_sampling_rate(self):
        self.assertEqual(self.vibration_analyzer.sampling_rate, 1)
        
    def test_dc_component(self):
        self.assertAlmostEqual(self.vibration_analyzer.x.mean(), 0.4407, places=4)
        self.assertAlmostEqual(self.vibration_analyzer.y.mean(), 2.048, places=4)
        self.assertAlmostEqual(self.vibration_analyzer.z.mean(), 0.9159, places=4)
        
    def test_fft(self):
        self.assertEqual(len(self.vibration_analyzer.x_fft), 5)
        self.assertAlmostEqual(self.vibration_analyzer.x_fft[0], 0.6955, places=4)
        self.assertAlmostEqual(self.vibration_analyzer.x_fft[-1], 0.6955, places=4)
        self.assertAlmostEqual(self.vibration_analyzer.y_fft[1], 0, places=4)
        self.assertAlmostEqual(self.vibration_analyzer.z_fft[3], 0.1155, places=4)
