# Shared pipeline for the revision experiments.
# Mirrors Notebooks A/B of the original study, with two changes that are switchable:
#   fold_encode=True  -> rare-country grouping and one-hot vocabulary fitted on the training part only
#   resample in {'smote','smotenc','none'}
import os, time, random, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from imblearn.over_sampling import SMOTE, SMOTENC
import xgboost as xgb
import lightgbm as lgb

SEED = 42
DATA = 'hotel_bookings.csv'
LEAK = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'booking_changes', 'agent', 'company']
CAT = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
       'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']
MODELS = ['Logistic Regression', '1D CNN', 'DNN', 'CNN+DNN Hybrid', 'XGBoost', 'LightGBM']

# configuration chosen most often across the five CV folds of the original study (Notebook A)
CFG = {
    'Logistic Regression': {'C': 1.0},
    '1D CNN': {'learning_rate': 0.0005, 'dropout': 0.4, 'width': 1.0, 'batch_size': 256},
    'DNN': {'learning_rate': 0.001, 'dropout': 0.4, 'width': 1.0, 'batch_size': 256},
    'CNN+DNN Hybrid': {'learning_rate': 0.001, 'dropout': 0.4, 'width': 1.0, 'batch_size': 256},
    'XGBoost': {'max_depth': 10, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.6,
                'min_child_weight': 1, 'reg_lambda': 0.5},
    'LightGBM': {'num_leaves': 127, 'learning_rate': 0.03, 'min_child_samples': 10, 'colsample_bytree': 0.6,
                 'subsample': 1.0, 'reg_lambda': 0.0},
}


def load(dedup=True):
    raw = pd.read_csv(DATA)
    df = raw.drop_duplicates().reset_index(drop=True) if dedup else raw.copy()
    df['_exact_group'] = df.groupby(list(raw.columns), dropna=False, sort=False).ngroup().values
    df = df.drop(columns=LEAK)
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['was_waitlisted'] = (df['days_in_waiting_list'] > 0).astype(int)
    df['is_family'] = ((df['children'] + df['babies']) > 0).astype(int)
    df = df[df['total_guests'] > 0].reset_index(drop=True)
    arrival = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-' +
                             df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
    booking = arrival - pd.to_timedelta(df['lead_time'], unit='D')
    EXACT_GROUPS[:] = [df.pop('_exact_group').values]
    return df, arrival, booking


EXACT_GROUPS = []


def group_ids(df):
    """rows with identical model inputs (after column removal) share a group id"""
    return df.drop(columns=['is_canceled']).groupby(list(df.columns.drop('is_canceled')), dropna=False, sort=False).ngroup().values


def exact_groups():
    """exact duplicates on all 32 original columns share a group id (from the last load())"""
    return EXACT_GROUPS[0]


def global_country(df):
    """original study: rare countries grouped on the full (deduplicated) data"""
    d = df.copy()
    vc = d['country'].value_counts()
    d.loc[d['country'].isin(vc[vc < 100].index), 'country'] = 'Other'
    return d


class Encoder:
    """rare-country grouping + one-hot encoding, fitted on training rows only"""
    def __init__(self, cat=CAT, rare=100, fit_rare=True):
        self.cat, self.rare, self.fit_rare = cat, rare, fit_rare

    def fit(self, X):
        if self.fit_rare:
            vc = X['country'].value_counts()
            self.keep = set(vc[vc >= self.rare].index)
        else:
            self.keep = None
        Xc = self._group(X)
        self.levels = {c: sorted(Xc[c].astype(str).unique()) for c in self.cat}
        self.num = [c for c in X.columns if c not in self.cat]
        return self

    def _group(self, X):
        X = X.copy()
        if self.keep is not None:
            X.loc[~X['country'].isin(self.keep), 'country'] = 'Other'
        return X

    def transform(self, X):
        X = self._group(X)
        parts = [X[self.num].astype('float32').reset_index(drop=True)]
        for c in self.cat:
            v = X[c].astype(str).values
            parts.append(pd.DataFrame({f'{c}_{l}': (v == l).astype('float32') for l in self.levels[c]}))
        return pd.concat(parts, axis=1)

    def codes(self, X):
        """integer codes (for SMOTENC); unseen level -> -1"""
        X = self._group(X).reset_index(drop=True)
        out = X[self.num].astype('float32').copy()
        for c in self.cat:
            m = {l: i for i, l in enumerate(self.levels[c])}
            out[c] = X[c].astype(str).map(m).fillna(-1).astype(int)
        return out

    def onehot_from_codes(self, C):
        parts = [C[self.num].astype('float32').reset_index(drop=True)]
        for c in self.cat:
            v = C[c].values
            parts.append(pd.DataFrame({f'{c}_{l}': (v == i).astype('float32') for i, l in enumerate(self.levels[c])}))
        return pd.concat(parts, axis=1)


def prepare_split(Xdf, y, tr_idx, te_idx, val_mode='random', val_order=None, fold_encode=True,
                  resample='smote', drop_cols=()):
    """Xdf: un-encoded features (cat columns as strings). Returns dict with arrays."""
    Xdf = Xdf.drop(columns=list(drop_cols))
    cat = [c for c in CAT if c in Xdf.columns]
    Xtr, Xte = Xdf.iloc[tr_idx].reset_index(drop=True), Xdf.iloc[te_idx].reset_index(drop=True)
    ytr, yte = y[tr_idx], y[te_idx]
    num = [c for c in Xdf.columns if c not in cat]
    up = Xtr[num].quantile(0.99)
    Xtr[num] = Xtr[num].clip(upper=up, axis=1)
    Xte[num] = Xte[num].clip(upper=up, axis=1)
    if val_mode == 'random':
        i_in, i_val = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=SEED)
    else:  # latest 10 % of the training period
        order = np.asarray(val_order)[tr_idx]
        cut = np.quantile(order.astype('int64'), 0.9)
        is_val = order.astype('int64') > cut
        i_in, i_val = np.where(~is_val)[0], np.where(is_val)[0]
    Xin, Xval, yin, yval = Xtr.iloc[i_in], Xtr.iloc[i_val], ytr[i_in], ytr[i_val]
    enc = Encoder(cat=cat, fit_rare=fold_encode)
    enc.fit(Xin if fold_encode else Xdf)   # global vocabulary when fold_encode=False
    sc = MinMaxScaler()
    def nums(A):
        return A
    if resample == 'smotenc':
        Cin = enc.codes(Xin)
        Cin[num] = sc.fit_transform(Cin[num])
        cat_idx = [Cin.columns.get_loc(c) for c in cat]
        sm = SMOTENC(categorical_features=cat_idx, random_state=SEED)
        Cs, ys = sm.fit_resample(Cin, yin)
        Xs = enc.onehot_from_codes(pd.DataFrame(Cs, columns=Cin.columns))
    else:
        Ein = enc.transform(Xin)
        Ein[num] = sc.fit_transform(Ein[num])
        if resample == 'smote':
            Xs, ys = SMOTE(random_state=SEED).fit_resample(Ein.values, yin)
            Xs = pd.DataFrame(Xs, columns=Ein.columns)
        else:
            Xs, ys = Ein, yin
    Ev, Et = enc.transform(Xval), enc.transform(Xte)
    Ev[num] = sc.transform(Ev[num])
    Et[num] = sc.transform(Et[num])
    return dict(Xtr=np.asarray(Xs, dtype='float32'), ytr=np.asarray(ys), Xval=Ev.values.astype('float32'), yval=yval,
                Xte=Et.values.astype('float32'), yte=yte, te_idx=np.asarray(te_idx), features=list(Et.columns),
                n_in=len(yin))


def _nn(name, n_feat, c):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, BatchNormalization, Concatenate, Reshape
    from tensorflow.keras.regularizers import l2
    w, dr = c['width'], c['dropout']
    if name == '1D CNN':
        return Sequential([Input(shape=(n_feat,)), Reshape((n_feat, 1)),
                           Conv1D(int(64 * w), 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
                           BatchNormalization(), Dropout(dr),
                           Conv1D(int(128 * w), 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
                           BatchNormalization(), Dropout(dr), Flatten(),
                           Dense(int(64 * w), activation='relu', kernel_regularizer=l2(1e-4)), Dropout(dr),
                           Dense(1, activation='sigmoid')])
    if name == 'DNN':
        return Sequential([Input(shape=(n_feat,)),
                           Dense(int(512 * w), activation='relu', kernel_regularizer=l2(1e-4)), BatchNormalization(), Dropout(dr),
                           Dense(int(256 * w), activation='relu', kernel_regularizer=l2(1e-4)), BatchNormalization(), Dropout(dr),
                           Dense(int(128 * w), activation='relu', kernel_regularizer=l2(1e-4)), BatchNormalization(), Dropout(dr),
                           Dense(int(64 * w), activation='relu', kernel_regularizer=l2(1e-4)), BatchNormalization(),
                           Dense(1, activation='sigmoid')])
    inp = Input(shape=(n_feat,))
    cb = Reshape((n_feat, 1))(inp)
    cb = Conv1D(int(64 * w), 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(cb)
    cb = BatchNormalization()(cb)
    cb = Conv1D(int(128 * w), 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(cb)
    cb = BatchNormalization()(cb)
    cb = Flatten()(cb)
    cb = Dense(int(128 * w), activation='relu', kernel_regularizer=l2(1e-4))(cb)
    db = Dense(int(256 * w), activation='relu', kernel_regularizer=l2(1e-4))(inp)
    db = BatchNormalization()(db)
    db = Dropout(dr)(db)
    db = Dense(int(128 * w), activation='relu', kernel_regularizer=l2(1e-4))(db)
    hb = Concatenate()([cb, db])
    hb = BatchNormalization()(hb)
    hb = Dropout(dr)(hb)
    hb = Dense(int(64 * w), activation='relu', kernel_regularizer=l2(1e-4))(hb)
    hb = Dropout(dr)(hb)
    return Model(inp, Dense(1, activation='sigmoid')(hb))


def fit_predict(name, S, cfg=None, seed_offset=0, class_weight=False):
    """train one model on split S, return (val_proba, test_proba)"""
    c = dict(CFG[name] if cfg is None else cfg)
    Xtr, ytr, Xv, yv, Xt = S['Xtr'], S['ytr'], S['Xval'], S['yval'], S['Xte']
    pos_w = (len(ytr) - ytr.sum()) / ytr.sum() if class_weight else 1.0
    if name == 'Logistic Regression':
        m = LogisticRegression(C=c['C'], max_iter=2000, random_state=SEED,
                               class_weight='balanced' if class_weight else None).fit(Xtr, ytr)
        return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
    if name == 'XGBoost':
        m = xgb.XGBClassifier(n_estimators=2000, tree_method='hist', eval_metric='logloss', early_stopping_rounds=50,
                              random_state=SEED, n_jobs=2, scale_pos_weight=pos_w, **c)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
    if name == 'LightGBM':
        m = lgb.LGBMClassifier(n_estimators=2000, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1,
                               scale_pos_weight=pos_w, **c)
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    net = _nn(name, Xtr.shape[1], c)
    net.compile(optimizer=Adam(learning_rate=c['learning_rate']), loss='binary_crossentropy')
    pat = (5, 2) if name == '1D CNN' else (7, 3)
    cw = {0: 1.0, 1: float(pos_w)} if class_weight else None
    net.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=50, batch_size=c['batch_size'], verbose=0, class_weight=cw,
            callbacks=[EarlyStopping(monitor='val_loss', patience=pat[0], restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', patience=pat[1], factor=0.5)])
    return (net.predict(Xv, batch_size=4096, verbose=0).ravel(), net.predict(Xt, batch_size=4096, verbose=0).ravel())


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred), 'F1': f1_score(y, pred), 'ROC-AUC': roc_auc_score(y, p),
            'PR-AUC': average_precision_score(y, p)}


def log(msg, path):
    with open(path, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


def ft_transformer(features, d=64, blocks=3, heads=8, att_drop=0.2, ffn_drop=0.1):
    """FT-Transformer (Gorishniy et al., 2021). Each numeric column becomes a token x*w+b; each categorical
    variable becomes one token = (one-hot group) @ E + b, i.e. an embedding lookup for valid one-hot rows."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from keras import ops
    num_idx = [i for i, f in enumerate(features) if not any(f.startswith(c + '_') for c in CAT)]
    groups = [[i for i, f in enumerate(features) if f.startswith(c + '_')] for c in CAT]
    groups = [g for g in groups if g]
    inp = Input(shape=(len(features),))

    class Tokenizer(layers.Layer):
        def build(self, _):
            self.wn = self.add_weight(shape=(len(num_idx), d), initializer='glorot_uniform', name='wn')
            self.bn = self.add_weight(shape=(len(num_idx), d), initializer='zeros', name='bn')
            self.E = [self.add_weight(shape=(len(g), d), initializer='glorot_uniform', name=f'e{j}') for j, g in enumerate(groups)]
            self.bc = self.add_weight(shape=(len(groups), d), initializer='zeros', name='bc')
            self.cls = self.add_weight(shape=(1, 1, d), initializer='glorot_uniform', name='cls')

        def call(self, x):
            xn = tf.gather(x, num_idx, axis=1)
            tn = xn[:, :, None] * self.wn[None] + self.bn[None]
            tc = tf.stack([tf.gather(x, g, axis=1) @ self.E[j] for j, g in enumerate(groups)], axis=1) + self.bc[None]
            cls = tf.repeat(self.cls, tf.shape(x)[0], axis=0)
            return tf.concat([cls, tn, tc], axis=1)

    h = Tokenizer()(inp)
    for b in range(blocks):
        a = layers.LayerNormalization()(h) if b > 0 else h
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads, dropout=att_drop)(a, a)
        h = h + a
        f = layers.LayerNormalization()(h)
        f = layers.Dense(int(d * 4 / 3) * 2)(f)
        f1, f2 = ops.split(f, 2, axis=-1)
        f = layers.Dropout(ffn_drop)(f1 * ops.relu(f2))       # ReGLU
        h = h + layers.Dense(d)(f)
    out = layers.Dense(1, activation='sigmoid')(layers.ReLU()(layers.LayerNormalization()(ops.take(h, 0, axis=1))))
    return Model(inp, out)


def fit_ft(S, seed_offset=0, lr=1e-3, batch=1024):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + seed_offset)
    net = ft_transformer(S['features'], d=32, blocks=2, heads=4)
    net.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5), loss='binary_crossentropy')
    net.fit(S['Xtr'], S['ytr'], validation_data=(S['Xval'], S['yval']), epochs=50, batch_size=batch, verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)])
    return net.predict(S['Xval'], batch_size=4096, verbose=0).ravel(), net.predict(S['Xte'], batch_size=4096, verbose=0).ravel()
