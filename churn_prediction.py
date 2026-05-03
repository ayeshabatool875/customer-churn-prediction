import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Generate realistic telecom customer data ───────────────────────────
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'customer_id'      : range(1, n + 1),
    'tenure_months'    : np.random.randint(1, 72, n),
    'monthly_charges'  : np.random.uniform(20, 120, n).round(2),
    'total_charges'    : np.random.uniform(100, 8000, n).round(2),
    'num_products'     : np.random.randint(1, 5, n),
    'support_calls'    : np.random.randint(0, 10, n),
    'payment_delays'   : np.random.randint(0, 6, n),
    'satisfaction_score': np.random.randint(1, 6, n),
    'contract_type'    : np.random.choice(
                            ['Month-to-Month', 'One Year', 'Two Year'],
                            n, p=[0.5, 0.3, 0.2]),
})

# Realistic churn logic
churn_prob = (
    0.3 * (df['tenure_months'] < 12).astype(int) +
    0.2 * (df['support_calls'] > 5).astype(int) +
    0.2 * (df['payment_delays'] > 2).astype(int) +
    0.15 * (df['satisfaction_score'] < 3).astype(int) +
    0.15 * (df['contract_type'] == 'Month-to-Month').astype(int)
)
df['churned'] = (churn_prob + np.random.uniform(0, 0.2, n) > 0.4).astype(int)

print(f"Dataset: {len(df)} customers | Churn rate: {df['churned'].mean():.1%}")

# ── 2. Feature Engineering ────────────────────────────────────────────────
df['contract_encoded'] = df['contract_type'].map(
    {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2})
df['avg_charge_per_month'] = df['total_charges'] / (df['tenure_months'] + 1)
df['risk_score'] = (df['support_calls'] * 0.3 +
                    df['payment_delays'] * 0.4 +
                    (5 - df['satisfaction_score']) * 0.3)

features = ['tenure_months', 'monthly_charges', 'num_products',
            'support_calls', 'payment_delays', 'satisfaction_score',
            'contract_encoded', 'avg_charge_per_month', 'risk_score']

X = df[features]
y = df['churned']

# ── 3. Train Models ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

rf  = RandomForestClassifier(n_estimators=100, random_state=42)
lr  = LogisticRegression(random_state=42, max_iter=1000)
rf.fit(X_train, y_train)
lr.fit(X_train_sc, y_train)

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:, 1])
print(f"Random Forest AUC : {rf_auc:.3f}")
print(f"Logistic Reg AUC  : {lr_auc:.3f}")

# ── 4. Dashboard ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Customer Churn Prediction Dashboard',
             fontsize=20, fontweight='bold')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# KPI cards
kpis = [
    ('Total Customers', f"{len(df):,}", '#3498db'),
    ('Churn Rate',      f"{df['churned'].mean():.1%}", '#e74c3c'),
    ('RF Model AUC',   f"{rf_auc:.3f}", '#2ecc71'),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_axes([0.05 + i * 0.32, 0.88, 0.27, 0.08])
    ax.set_facecolor(color)
    ax.text(0.5, 0.6,  value, transform=ax.transAxes,
            ha='center', fontsize=16, fontweight='bold', color='white')
    ax.text(0.5, 0.15, label, transform=ax.transAxes,
            ha='center', fontsize=9,  color='white')
    ax.set_xticks([]); ax.set_yticks([])

# Feature importance
ax1 = fig.add_subplot(gs[0, 0])
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
colors_bar = ['#e74c3c' if v > imp.median() else '#3498db'
              for v in imp.values]
imp.plot(kind='barh', ax=ax1, color=colors_bar)
ax1.set_title('Feature Importance', fontweight='bold')
ax1.set_xlabel('Importance Score')
ax1.grid(True, alpha=0.3, axis='x')

# ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
for model, X_t, label, color in [
        (rf, X_test,    f'Random Forest (AUC={rf_auc:.2f})', '#e74c3c'),
        (lr, X_test_sc, f'Log Regression (AUC={lr_auc:.2f})', '#3498db')]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_t)[:, 1])
    ax2.plot(fpr, tpr, label=label, color=color, linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_title('ROC Curve Comparison', fontweight='bold')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, rf.predict(X_test))
im = ax3.imshow(cm, cmap='Blues')
ax3.set_title('Confusion Matrix\n(Random Forest)', fontweight='bold')
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Stayed', 'Churned'])
ax3.set_yticklabels(['Stayed', 'Churned'])
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black',
                 fontsize=14, fontweight='bold')

# Churn by contract type
ax4 = fig.add_subplot(gs[1, 0])
ct = df.groupby('contract_type')['churned'].mean() * 100
bars = ax4.bar(ct.index, ct.values,
               color=['#e74c3c', '#f39c12', '#2ecc71'],
               edgecolor='white')
ax4.set_title('Churn Rate by Contract Type', fontweight='bold')
ax4.set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, ct.values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=9)
plt.setp(ax4.get_xticklabels(), rotation=10, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# Risk distribution
ax5 = fig.add_subplot(gs[1, 1])
churned     = df[df['churned'] == 1]['risk_score']
not_churned = df[df['churned'] == 0]['risk_score']
ax5.hist(not_churned, bins=20, alpha=0.6,
         color='#2ecc71', label='Stayed',   edgecolor='white')
ax5.hist(churned,     bins=20, alpha=0.6,
         color='#e74c3c', label='Churned',  edgecolor='white')
ax5.set_title('Risk Score Distribution', fontweight='bold')
ax5.set_xlabel('Risk Score')
ax5.set_ylabel('Number of Customers')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Top 10 at-risk customers
ax6 = fig.add_subplot(gs[1, 2])
df['churn_probability'] = rf.predict_proba(X)[:, 1]
top_risk = df.nlargest(10, 'churn_probability')[
    ['customer_id', 'churn_probability', 'monthly_charges']]
colors_risk = ['#e74c3c' if p > 0.7 else '#f39c12'
               for p in top_risk['churn_probability']]
bars = ax6.barh(
    [f"Customer {i}" for i in top_risk['customer_id']],
    top_risk['churn_probability'] * 100,
    color=colors_risk)
ax6.set_title('Top 10 At-Risk Customers', fontweight='bold')
ax6.set_xlabel('Churn Probability (%)')
ax6.axvline(x=70, color='red', linestyle='--',
            alpha=0.5, label='High Risk (70%)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='x')

plt.savefig('churn_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTOP 10 AT-RISK CUSTOMERS:")
print(top_risk.to_string(index=False))
print(f"\nModel ready — {(rf.predict(X_test)==y_test).mean():.1%} accuracy")
