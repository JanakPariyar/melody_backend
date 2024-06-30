import pandas as pd

# Load CSV file with song links and names
df = pd.read_csv("A:\\updated_song_link.csv")

def get_initial_song():
    random_row = df.sample(n=1).iloc[0]
    song_name = random_row['song_name']
    if pd.isnull(song_name) or song_name == 'NaN':
        song_name = 'surprise'
    
    return {song_name:'now your turn'}

def validate_user_song(initial_song, user_song):
    last_letter = initial_song[-1].lower()
    first_letter = user_song[0].lower()

    if last_letter != first_letter:
        return False, "The song does not start with the correct letter."

    if user_song not in df['song_name'].values:
        return False, "The song is not in the database. Please provide the song link."

    return True, "Valid song."

def add_song_to_db(song_name, song_url):
    new_row = {'song_name': song_name, 'URLS': song_url}
    global df
    df = df.append(new_row, ignore_index=True)
    df.to_csv("A:\\updated_song_link.csv", index=False)
    return "URL added successfully!"
