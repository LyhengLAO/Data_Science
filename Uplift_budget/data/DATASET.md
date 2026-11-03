# Dataset Hillstrom — E-Mail Marketing Uplift

## Présentation

Le dataset **Hillstrom** provient du *MineThatData E-Mail Analytics Challenge* (Kevin Hillstrom, 2008). Il décrit un **test A/B marketing** mené par une enseigne de vente en ligne (produits homme et femme) sur **64 000 clients**.

Les clients ont été répartis aléatoirement en 3 groupes (environ 1/3 chacun) :

| Groupe (`segment`) | Traitement reçu |
|---|---|
| `Mens E-Mail` | E-mail promotionnel collection homme |
| `Womens E-Mail` | E-mail promotionnel collection femme |
| `No E-Mail` | Aucun e-mail (groupe de contrôle) |

Pendant les 2 semaines suivant l'envoi, on a mesuré si chaque client a visité le site, acheté, et combien il a dépensé.

L'assignation étant aléatoire, la différence de résultats entre groupes traités et contrôle correspond à un **effet causal** de l'e-mail. C'est ce qui permet d'estimer l'**uplift** :

```
uplift(x) = P(conversion = 1 | e-mail, x) − P(conversion = 1 | pas d'e-mail, x)
```

## Colonnes

### Variables explicatives (features)

| Colonne | Type | Valeurs | Description |
|---|---|---|---|
| `recency` | entier | 1 à 12 | Mois écoulés depuis le dernier achat |
| `history_segment` | catégorielle ordinale | `1) $0 - $100` … `7) $1,000+` | Tranche de dépenses de l'année précédente |
| `history` | numérique | en $ | Montant exact dépensé l'année précédente |
| `mens` | binaire | 0 / 1 | A acheté des articles homme l'année précédente |
| `womens` | binaire | 0 / 1 | A acheté des articles femme l'année précédente |
| `zip_code` | catégorielle | `Urban`, `Suburban`, `Rural` | Type de zone de résidence |
| `newbie` | binaire | 0 / 1 | Nouveau client (inscrit dans les 12 derniers mois) |
| `channel` | catégorielle | `Phone`, `Web`, `Multichannel` | Canal d'achat de l'année précédente |

### Traitement

| Colonne | Type | Valeurs | Description |
|---|---|---|---|
| `segment` | catégorielle | `Mens E-Mail`, `Womens E-Mail`, `No E-Mail` | Groupe assigné lors du test A/B |

### Variables cibles (outcomes)

| Colonne | Type | Valeurs | Description |
|---|---|---|---|
| `visit` | binaire | 0 / 1 | A visité le site dans les 2 semaines suivantes |
| `conversion` | binaire | 0 / 1 | A acheté dans les 2 semaines suivantes |
| `spend` | numérique | en $ | Montant dépensé dans les 2 semaines suivantes |
