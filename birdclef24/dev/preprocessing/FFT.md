# Preprocessing

In audio signal processing, understanding frequency domain images and time domain images is very important for analyzing and modifying audio data. These two representations provide different perspectives on the sound signal:

### Time Domain Images
Time domain images are the most direct and common representation of audio signals. In the time domain, the image shows the amplitude (or pressure level) of the sound waveform changing over time.

**How ​​to view a time domain image:**
- **X-axis**: represents time. Usually, the time unit can be seconds or milliseconds.
- **Y-axis**: represents the amplitude of the sound. For digital audio, this is usually expressed as the sample value of the waveform; in the analog or physical world, it may represent the sound pressure level.

**Interpreting time domain images:**
- **The height (amplitude) of the waveform**: represents the loudness of the sound. The larger the amplitude, the louder the sound.
- **The density (frequency) of the waveform**: Fast up and down fluctuations represent high-frequency sounds (such as the sound of a soprano or a small bell), while slow fluctuations represent low-frequency sounds (such as a bass or drums).

Time domain images are great for observing the rhythm, loudness, and duration of sounds.

### Frequency Domain Images
Frequency domain images represent the intensity or energy of each frequency component contained in a sound. This image is usually obtained by applying a Fourier transform (such as FFT) to a time domain signal, revealing the spectral structure of the sound.

**How ​​to view frequency domain images:**
- **X-axis**: Represents frequency, usually in Hertz (Hz). Low frequencies are on the left and high frequencies are on the right.
- **Y-axis**: Usually represents the logarithm of amplitude (e.g., decibels dB), representing the energy or intensity of each frequency component.

**Interpreting frequency domain images:**
- **Peaks**: Peaks in the image indicate that the sound component of a particular frequency is particularly strong. For example, fundamental tones and harmonics in music usually form obvious peaks in the frequency domain image.
- **Broadband/Narrowband Features**: Broadband features (such as white noise) appear as relatively flat areas in the frequency domain, while narrowband features (such as a single note or tone) appear as sharp peaks.

Frequency domain images are particularly suitable for analyzing the pitch, timbre, and harmonic content of sounds, which is very useful in processing music, speech analysis, and noise control.

### Summary
Time domain images provide an intuitive view of how sounds change over time, suitable for analyzing rhythm and dynamic changes. Frequency domain images reveal the frequency components of sounds, suitable for analyzing pitch and texture of sounds. Depending on the problem you need to solve and the purpose of the analysis, you can choose the appropriate domain for viewing and processing.