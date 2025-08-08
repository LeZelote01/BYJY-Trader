# 💾 BYJY-Trader - Procédures de Sauvegarde & Restauration

## **Vue d'Ensemble du Système Backup**

Le système de backup du BYJY-Trader assure la protection complète des données avec une robustesse de niveau entreprise.

## **🔧 Configuration Backup**

### **Variables d'Environnement**
```bash
# .env
BACKUP_ENABLED=true           # Activer/désactiver les backups
BACKUP_INTERVAL=3600         # Intervalle en secondes (1h)
BACKUP_RETENTION_DAYS=30     # Rétention en jours
BACKUP_COMPRESSION=true      # Compression gzip
```

### **Structure des Répertoires**
```
📁 backups/
├── 📁 automatic/           # Backups automatiques
│   ├── 🗜️ backup_2025-08-05_15-30-00.db.gz
│   └── 🗜️ backup_2025-08-05_14-30-00.db.gz
├── 📁 manual/             # Backups manuels
│   └── 🗜️ backup_manual_migration.db.gz
└── 📁 temp/              # Fichiers temporaires
```

## **⚡ Procédures Automatiques**

### **Backup Automatique**
Le système crée automatiquement des sauvegardes selon l'intervalle configuré :

1. **Déclenchement** : Timer configurable (défaut 1h)
2. **Processus** :
   - Verrouillage transaction
   - Copie cohérente de la base
   - Compression gzip (-6 ratio optimal)
   - Validation intégrité
   - Nettoyage anciens backups

3. **Monitoring** : Logs détaillés + notifications erreurs

### **Nettoyage Automatique**
- **Rétention** : Configurable (défaut 30 jours)
- **Politique** : FIFO (Premier créé = Premier supprimé)
- **Sécurité** : Conservation minimum 3 backups

## **🔨 Procédures Manuelles**

### **1. Créer Backup Manuel**

```python
from core.backup_manager import get_backup_manager

# Créer backup avec nom personnalisé
backup_manager = get_backup_manager()
success = backup_manager.create_backup("migration_v1_2")

if success:
    print("✅ Backup créé avec succès")
else:
    print("❌ Échec création backup")
```

```bash
# Via CLI
python launcher/main.py backup create --name "migration_v1_2"
```

### **2. Lister les Backups**

```python
# Lister tous les backups
backups = backup_manager.list_backups()

for backup in backups:
    print(f"📋 {backup['name']} - {backup['size_mb']}MB - {backup['created_at']}")
```

```bash
# Via CLI
python launcher/main.py backup list
```

### **3. Restaurer un Backup**

⚠️ **ATTENTION** : La restauration remplace complètement la base existante !

```python
# Restauration avec vérification
backup_name = "backup_2025-08-05_15-30-00.db.gz"
success = backup_manager.restore_from_backup(backup_name)

if success:
    print("✅ Restauration réussie")
    # Redémarrer l'application
else:
    print("❌ Échec restauration")
```

```bash
# Via CLI avec confirmation
python launcher/main.py backup restore --name "backup_2025-08-05_15-30-00.db.gz" --confirm
```

## **🔍 Validation & Tests d'Intégrité**

### **Tests Automatiques**
Chaque backup est automatiquement testé :

1. **Test Compression** : Vérification intégrité gzip
2. **Test SQL** : Ouverture et requête test
3. **Test Schéma** : Validation structure tables
4. **Test Données** : Comptage enregistrements

### **Tests Manuels**
```python
# Test intégrité backup spécifique
integrity_ok = backup_manager.verify_backup_integrity("backup_name.db.gz")
print(f"🔍 Intégrité : {'✅ OK' if integrity_ok else '❌ ERREUR'}")
```

## **⚡ Performance & Optimisation**

### **Métriques de Performance**
- **Backup 100MB** : ~30 secondes
- **Compression** : Ratio 4:1 moyenne
- **Restore 100MB** : ~15 secondes
- **Validation** : ~5 secondes

### **Optimisations**
1. **Compression Level 6** : Balance vitesse/ratio
2. **Backup Incrémental** : Planifié v1.3
3. **Parallélisation** : Backup + validation simultanés
4. **SSD Storage** : Recommandé pour /backups

## **🚨 Gestion d'Erreurs**

### **Scénarios de Récupération**

#### **1. Base Corrompue**
```bash
# Backup automatique le plus récent
python launcher/main.py backup restore --latest --confirm

# Ou backup manuel spécifique
python launcher/main.py backup restore --name "backup_manual_good.db.gz"
```

#### **2. Migration Échouée**
```bash
# Rollback via backup pré-migration
python launcher/main.py backup restore --name "backup_pre_migration.db.gz"
```

#### **3. Perte de Données**
```bash
# Restore + validation
python launcher/main.py backup restore --name "backup_before_incident.db.gz"
python launcher/main.py database validate
```

### **Logs d'Erreur**
```python
# Monitoring backup status
import logging
logger = logging.getLogger('backup')

# Toutes les opérations sont loggées
logger.info("Backup started")
logger.error("Backup failed: disk full")  
logger.info("Restore completed successfully")
```

## **📊 Monitoring & Alertes**

### **Métriques Clés**
- ✅ **Backup Success Rate** : 100% attendu
- ⏱️ **Backup Duration** : Trend monitoring
- 💾 **Storage Usage** : Alerte si > 80%
- 🔍 **Integrity Checks** : Tous verts

### **Alertes Configurées**
1. **Backup Failed** : Alert immédiate
2. **Storage Full** : Alert 80% usage  
3. **Integrity Error** : Alert critique
4. **Restore Needed** : Alert opérationnelle

## **🔄 Intégration CI/CD**

### **Tests Automatisés**
```bash
# Tests backup dans pipeline
pytest tests/test_backup_complete.py -v
```

### **Backup Pré-Déploiement**
```bash
# Backup automatique avant migration
python launcher/main.py backup create --name "pre_deploy_$(date +%Y%m%d_%H%M)"
```

## **📋 Checklist Maintenance**

### **Quotidienne**
- [ ] Vérifier backup dernières 24h
- [ ] Contrôler espace disque
- [ ] Vérifier logs erreurs

### **Hebdomadaire**  
- [ ] Test restore backup aléatoire
- [ ] Vérification intégrité échantillon
- [ ] Nettoyage logs anciens

### **Mensuelle**
- [ ] Test restore complet environnement test
- [ ] Révision politique rétention
- [ ] Mise à jour documentation
- [ ] Formation équipe procédures urgence

## **🆘 Procédures d'Urgence**

### **Recovery d'Urgence (RTO < 15min)**
```bash
# 1. Identifier dernier backup valide
python launcher/main.py backup list --recent 5

# 2. Restore immédiat
python launcher/main.py backup restore --name "last_good_backup.db.gz" --emergency

# 3. Validation rapide
python launcher/main.py database quick-check

# 4. Redémarrage application
python launcher/main.py restart
```

### **Contacts d'Urgence**
- **Admin DB** : admin@byjy-trader.com  
- **DevOps** : devops@byjy-trader.com
- **Support** : support@byjy-trader.com