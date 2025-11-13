// --- Define your sensor pin ---
const int ECGSensorPin = A0;

// ----- Filter Settings -----

// --- Low-Pass Filter (removes "fuzz") ---
// This is the same EMA filter from before
const float alpha = 0.2; // Smoothing factor (0.01 to 0.2)
float lowPassValue = 0;

// --- High-Pass Filter (removes "baseline wander") ---
// We will use a simple high-pass filter:
// HighPass = RawSignal - LowPass
// This works because the low-pass signal is the "baseline"
// We then add a "midpoint" (512) to keep the signal centered.
float highPassValue = 0;

void setup() {
    Serial.begin(9600);
    
    pinMode(10, INPUT); // LO +
    pinMode(11, INPUT); // LO -

    // Initialize the low-pass filter with the first reading
    lowPassValue = analogRead(ECGSensorPin);
}

void loop() {
    if ((digitalRead(10) == 1) || (digitalRead(11) == 1)) {
        Serial.println('!');
    } else {
        // --- 1. Get the raw signal ---
        int rawValue = analogRead(ECGSensorPin);

        // --- 2. Apply Low-Pass Filter (removes "fuzz") ---
        // This is the same as before
        lowPassValue = (alpha * rawValue) + ((1.0 - alpha) * lowPassValue);
        
        // --- 3. Apply High-Pass Filter (removes "drift") ---
        // We subtract the slow-moving low-pass value (the "baseline")
        // from the raw signal to remove the drift.
        // We add 512 to re-center the signal in the middle of the
        // analog range (0-1023) for plotting.
        highPassValue = rawValue - lowPassValue + 512;
        
        // --- 4. Print the final, clean signal ---
        Serial.println(highPassValue);
    }
    
    delay(2);
}
