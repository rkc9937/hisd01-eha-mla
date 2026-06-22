# EHA Policy Diffusion

TLDR; for coefficents look at refinement_6_eha_logit_coefficients.csv and
kentucky_refinement_6_eha_logit_coefficients.csv

## Goal

The goal of this repository is to measure the effects of the **learning mechanism** on the likelihood of policy diffusion.

## Mechanisms of Policy Diffusion

The literature on policy diffusion has identified 4 main mechanisms of why a policy would diffuse.

1. Learning
2. Coercion
3. Imitation
4. Competition

It is ideal that a government **learns** from another government when adopting one of their policies. Usually a policy is learned by some government when it is successful in another unit of government.

### Quantifying the Learning Mechanism

Since we cannot (or it is very hard to) directly measure how successful a policy is, researchers measure how much opprutunity the policy has to be learned as a direct substitue for evidence of learning.

"Policymakers cannot learn about policies that have not yet been tried" (Shipan & Voldon, 2008)

Thus our learning variable is **Proporation of State Population with Local Restriction** or the proportion of the population in muncipalities that have the ordinace to the total population of the containing government.

## Smoke Free Ordinance Study

To parallel a part of Shipan & Voldon's study, we are looking at the diffusion of smoke free policies from city-to-city (horizontal diffusion).

Shipan & Voldon look at anti-smoking policies in restauraunts, and government buildings, while my analysis looks at municipalities that passed 100% smoke free policies.

## Modelling policy diffusion as EHA

Event History Analysis (EHA), or Surivival Analysis, aims to find a model that allows us to get the likelihood of when an event happens.

This specifically is called the hazard, h(t) = Pr("an event happens exactly at time t").

In our case, the hazard represents the "probability a policy diffuses at time t".

We treat each time interval as a seperate binary observation, and this way we can fit a logit model and see the impact each of our covariates have.

Shipan & Voldon look at the magnitude and sign of the learning variable to determine success.

A X percent increase in the learning mechanism coefficent corresponds to an $$e^[X]$$ percent change in odds to adption.

## Covariates

These are factors that would influence the odds of the hazard. I used almost all of Shipan & Voldon's covariates.

1. Local Population
2. Percent High School Grads
3. Per Capita Income
4. Per Capita Government Spending
5. Percent White
6. Percent Smokers

> [!NOTE]
> Currently wokring on adding Tobacco Lobbyists and State Government Ideology

## Kentucky Obervations

This time, I looked at municipalities in Kentucky and ran the same analysis as I did to Texas.

There were three models that I came up with, due to partial data:

1.Complete

This model was the same as the Texas study in terms of covariates.

2.Primary

This model included an additonal covariate that Shipan & Voldon included which was the **Proportion of Tobacco Lobbyists**.

The data is incomplete (we have the total number of lobbyist per year, but not the number of tobacco lobysists, i.e. we have the denominator), so the primary model **forward-fills** the numerator from previous observed ratios.

3.Sensitive

This model includes all the Primary Models, but backfills the numerator for the **Proportion of Tobacco Lobbyists**.

It also includes the same **State Government Ideology** which forward fills from 2017-2022.