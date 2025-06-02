import os

import pandas as pd


def ASVspoof2019_LA():
    root = '/path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols'
    files = [
        'ASVspoof2019.LA.cm.train.trl.txt',
        'ASVspoof2019.LA.cm.dev.trl.txt',
        'ASVspoof2019.LA.cm.eval.trl.txt'
    ]
    dir = 'data/metadata'
    dirs_new = [
        'ASVspoof2019_LA_train',
        'ASVspoof2019_LA_dev',
        'ASVspoof2019_LA_eval'
    ]
    for i, file in enumerate(files):
        with open(os.path.join(root, file), 'r') as f:
            lines = f.readlines()
        df = pd.DataFrame()
        for line in lines:
            line = line.strip().split(' ')
            speaker = line[0]
            file = line[1]
            label = line[-1]
            attack = line[-2]
            file_path = f'data/audio/{dirs_new[i]}/flac/{file}.flac'
            df = df.append({'file': file_path,
                            'label': label,
                            'speaker': speaker,
                            'attack': attack},
                           ignore_index=True)
        df.to_csv(os.path.join(dir, f'{dirs_new[i]}.csv'), index=False)


def ASVspoof2021_LA():
    file = '/path/to/ASVspoof2021/keys/LA/CM/trial_metadata.txt'
    file_new = 'data/metadata/ASVspoof2021_LA_eval.csv'
    with open(file, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame()
    for line in lines:
        line = line.strip().split(' ')
        file = line[1]
        label = line[5]
        speaker = line[0]
        codec = line[2]
        trans = line[3]
        phase = line[-1]
        if phase != 'eval':
            continue
        file_path = f'data/audio/ASVspoof2021_LA_eval/wav/{file}.wav'
        df = df.append({'file': file_path,
                        'label': label,
                        'speaker': speaker,
                        'codec': codec,
                        'trans': trans},
                       ignore_index=True)
    df.to_csv(file_new, index=False)


def ASVspoof2021_DF():
    file = '/path/to/ASVspoof2021/keys/DF/CM/trial_metadata.txt'
    file_new = 'data/metadata/ASVspoof2021_DF_eval.csv'
    with open(file, 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame()
    for line in lines:
        line = line.strip().split(' ')
        file = line[1]
        label = line[5]
        file_path = f'data/audio/ASVspoof2021_DF_eval/wav/{file}.wav'
        df = df.append({'file': file_path,
                        'label': label},
                       ignore_index=True)
    df.to_csv(file_new, index=False)


def In_The_Wild():
    file = '/Database/Fraunhofer_Deepfake_in_wild/meta.csv'
    file_new = 'data/metadata/In_The_Wild.csv'
    df = pd.read_csv(file)
    df_new = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        file_path = f'data/audio/In_The_Wild/wav16k/{row["file"]}'
        df_new = df_new.append({'file': file_path,
                                'label': row['label']},
                               ignore_index=True)
    df_new.to_csv(file_new, index=False)


def FakeOrReal():
    root = '/path/to/audio/FakeOrReal'
    dirs = ['testing', 'validation']
    df = pd.DataFrame()
    for dir in dirs:
        curr_dir = os.path.join(root, dir)
        for label in os.listdir(curr_dir):
            curr_subdir = os.path.join(curr_dir, label)
            for file in os.listdir(curr_subdir):
                file_path = os.path.join(curr_subdir, file)
                df = df.append({'file': file_path,
                                'label': label},
                               ignore_index=True)
    df.to_csv('data/metadata/FakeOrReal.csv', index=False)


def WaveFake(ratio=0.4):
    audio = 'data/audio/WaveFake'
    metadata = '/path/to/WaveFake/protocol.txt'
    df = pd.read_csv(metadata, sep=' ', header=None)
    values = df[0].value_counts()
    counts = {k: int(v * ratio) for k, v in zip(values.index, values.values)}
    print(counts)
    df_new = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        type = row[0]
        if counts[type] == 0:
            break
        file = row[1] + '.wav'
        label = row[5]
        file_path = os.path.join(audio, file)
        df_new = df_new.append({'file': file_path,
                                'label': label},
                               ignore_index=True)
        counts[type] -= 1
    df_new.to_csv('data/metadata/WaveFake.csv', index=False)


def WaveFake_part(part='JSUT', real=4971, fake=10000):
    audio = 'data/audio/WaveFake'
    metadata = '/path/to/WaveFake/protocol.txt'
    df = pd.read_csv(metadata, sep=' ', header=None)
    # values = df[0].value_counts()
    # counts = int(values[part] * ratio)
    df = df[df[0] == part]
    # df = df.sample(n=counts, random_state=42)
    df_real = df[df[5] == 'bonafide'].sample(n=real, random_state=42)
    df_fake = df[df[5] == 'spoof'].sample(n=fake, random_state=42)
    df = pd.concat([df_real, df_fake], ignore_index=True)
    df_new = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        file = row[1] + '.wav'
        label = row[5]
        file_path = os.path.join(audio, file)
        df_new = df_new.append({'file': file_path,
                                'label': label,
                                'origin': part},
                               ignore_index=True)
    df_new.to_csv(f'data/metadata/WaveFake_{part}.csv', index=False)


def ADD2022_dev():
    audio = 'data/audio/ADD2022_dev'
    metadata = '/path/to/ADD2022/ADD_train_dev/label/dev_label.txt'
    df = pd.read_csv(metadata, sep=' ', header=None)
    df_new = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        file = row[0]
        label = row[1]
        file_path = os.path.join(audio, file)
        df_new = df_new.append({'file': file_path,
                                'label': label},
                               ignore_index=True)
    df_new.to_csv('data/metadata/ADD2022_dev.csv', index=False)


def ADD2022_eval():
    audio = 'data/audio/ADD2022_eval'
    metadata = '/path/to/ADD_metadata/track3_R2_label.txt'
    df = pd.read_csv(metadata, sep=' ', header=None)
    df_new = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        file = row[0]
        label = row[1]
        attack = int(row[2])
        file_path = os.path.join(audio, file)
        df_new = df_new.append({'file': file_path,
                                'label': label,
                                'attack': attack},
                               ignore_index=True)
    df_new.to_csv('data/metadata/ADD2022_eval.csv', index=False)


def SpoofCeleb():
    root = '/path/to/SpoofCeleb/metadata'
    parts = ['development', 'evaluation']
    for part in parts:
        ori_file = os.path.join(root, f'{part}.csv')
        df = pd.read_csv(ori_file)
        df_new = pd.DataFrame()
        for i in range(len(df)):
            row = df.iloc[i]
            file = row[0]
            file = f'data/audio/SpoofCeleb/{part}/{file}'
            speaker = row[1]
            attack = row[2]
            label = 'bonafide' if attack == 'a00' else 'spoof'
            df_new = df_new.append({'file': file,
                                    'label': label,
                                    'speaker': speaker,
                                    'attack': attack},
                                   ignore_index=True)
        df_new.to_csv(f'data/metadata/SpoofCeleb_{part}.csv', index=False)


def select_dev(file, num=10000):
    df = pd.read_csv(file)
    df = df.sample(n=num, random_state=42)
    file_name = file.split('/')[-1].split('.')[0] + '_dev.csv'
    file_path = os.path.join('data/metadata', file_name)
    df.to_csv(file_path, index=False)


def merge_metadata(file1, file2, save_path):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    ASVspoof2019_LA()
    ASVspoof2021_LA()
    ASVspoof2021_DF()
    In_The_Wild()
    select_dev('data/metadata/In_The_Wild.csv')
    FakeOrReal()
    WaveFake()
    WaveFake_part('JSUT', ratio=1.0)
    WaveFake_part('LJSPEECH', real=4971, fake=10000)
    merge_metadata('data/metadata/WaveFake_JSUT.csv',
                   'data/metadata/WaveFake_LJSPEECH.csv',
                   'data/metadata/WaveFake.csv')
    ADD2022_eval()
    SpoofCeleb()
