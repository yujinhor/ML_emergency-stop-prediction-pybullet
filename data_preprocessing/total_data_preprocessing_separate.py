import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# ====================================================
# 1. 파일 경로 및 설정 (Configuration)
# ====================================================
base_path = os.path.dirname(os.path.abspath(__file__))

# 학습용 파일
train_steps_path = os.path.join(base_path, 'train_steps.csv')
train_summary_path = os.path.join(base_path, 'train_summary.csv')

# 태스트 파일
test_steps_path = os.path.join(base_path, 'val_steps.csv')
test_summary_path = os.path.join(base_path, 'val_summary.csv')

# 설정값
target_col = 'result_is_failure'  # 정답 컬럼 이름
SEQ_LEN = 2400                    # 시퀀스 길이, 실패 데이터에서 제일 긴 데이터가 2359였으므로 다 가져가기로..
feature_cols = [                  # 입력으로 쓸 변수들
    'speed', 'dist_to_wall', 'drag_force_N', 
    'mass_kg', 'friction_cond', 'air_density', 'trigger_dist_m', 'brake_torque', 'init_speed_cmd'
]
static_cols_to_merge = [          # surmmary 파일에서 가져올 물리 변수들
    'episode', 'mass_kg', 'friction_cond', 'air_density', 'brake_torque', 'init_speed_cmd'
]

# ====================================================
# 2. 데이터 로드 및 병합 함수 (Load & Merge Function)
# ====================================================
def load_and_merge(steps_path, summary_path, mode="Train"):
    print(f"\n[{mode}] 데이터 로드 중...")
    if not os.path.exists(steps_path) or not os.path.exists(summary_path):
        print(f"⚠️ {mode} 파일이 없습니다. 경로를 확인하세요.")
        return None

    df_steps = pd.read_csv(steps_path)
    df_summary = pd.read_csv(summary_path)
    


    # 병합
    df_merged = pd.merge(df_steps, df_summary[static_cols_to_merge], on='episode', how='left')
    
    # 중복 제거
    if f'{target_col}_y' in df_merged.columns:
        df_merged = df_merged.rename(columns={f'{target_col}_y': target_col})
        if f'{target_col}_x' in df_merged.columns:
            df_merged = df_merged.drop(columns=[f'{target_col}_x'])

    print(f"[{mode}] 병합 완료: {df_merged.shape}")
    return df_merged

# Train, Test 각각 로드
train_df = load_and_merge(train_steps_path, train_summary_path, "Train")
test_df = load_and_merge(test_steps_path, test_summary_path, "Test")

# ====================================================
# 3. 스케일링 (Scaling) 
# ====================================================
scaler = MinMaxScaler()

# 1. Train 데이터로 기준 만든다. 
scaler.fit(train_df[feature_cols])
print("\n Scaler 학습 완료 (Train 데이터 기준)")

# 2. 그 자로 Train 데이터를 잽니다. (Transform)
train_df[feature_cols] = scaler.transform(train_df[feature_cols])

# 3. 같은 기준으로 Test 데이터 스케일링. (Transform only)
if test_df is not None:
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    print("Test 데이터 변환 완료 (Train 기준으로 스케일링됨)")

# ====================================================
# 4. 3차원 텐서 변환 (Windowing)
# ====================================================
def create_sequences(df, seq_len):
    if df is None: return None, None
    
    episode_ids = df['episode'].unique()
    X_list = []
    y_list = []
    
    for ep in episode_ids:
        group = df[df['episode'] == ep]
        features = group[feature_cols].values
        
        # 정답 라벨 (에피소드 마지막 값)
        label = group[target_col].iloc[-1]
        
        # Windowing (Tail Truncation & Padding)
        if len(features) >= seq_len:
            features = features[-seq_len:, :]
        else:
            pad_len = seq_len - len(features)
            features = np.pad(features, ((0 ,pad_len), (0, 0)), mode='constant')
            
        X_list.append(features)
        y_list.append(label)
        
    return np.array(X_list), np.array(y_list)

# 변환 실행
print("\n[Tensor 변환 중...]")
X_train, y_train = create_sequences(train_df, SEQ_LEN)
X_test, y_test = create_sequences(test_df, SEQ_LEN)

# ====================================================
# 5. 데이터 저장 (Save)
# ====================================================
save_dir = os.path.join(base_path, 'processed_datas')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"\n[저장 시작] 경로: {save_dir}")

'''# 엑셀(CSV) 확인용 파일 저장
# 스케일링 된 값들과 라벨(result_is_failure)이 모두 들어있는 표 데이터.
train_csv_path = os.path.join(save_dir, 'train_data_scaled.csv')
train_df.to_csv(train_csv_path, index=False)
print(f"[엑셀용] Train 데이터(라벨 포함) 저장 완료: {train_csv_path}")

if test_df is not None:
    test_csv_path = os.path.join(save_dir, 'test_data_scaled.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"[엑셀용] Test 데이터(라벨 포함) 저장 완료: {test_csv_path}")'''



# Train 저장
np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
print(f"📌 Train 텐서 저장 완료: {X_train.shape}")

# Test 저장 (파일이 있었을 경우만)
if X_test is not None:
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    print(f"📌 Test 텐서 저장 완료: {X_test.shape}")

print("-" * 50)
print("끝!")