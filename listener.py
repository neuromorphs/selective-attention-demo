from pylsl import StreamInlet, resolve_stream

def main():
    # First, resolve an EEG stream on the Lab network
    # Replace 'EEG' with the type of data you're expecting, if different
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # Create a new inlet to receive data from the stream
    inlet = StreamInlet(streams[0])

    try:
        while True:
            # Get a new sample (you can also omit the timestamp part if not needed)
            sample, timestamp = inlet.pull_sample()
            print(f"Timestamp: {timestamp} Sample: {sample}")
    except KeyboardInterrupt:
        print("\nStream listening ended by user. Closing stream...")
        inlet.close_stream()
        print("Stream closed.")

if __name__ == "__main__":
    main()
