import csv
import random
import datetime
import argparse
import numpy as np


def convert_km(scale_factor):
    if scale_factor % 1000000 == 0:
        return str(int(scale_factor / 1000000)) + 'M'

    if scale_factor % 1000 == 0:
        return str(int(scale_factor / 1000)) + 'K'

    return str(scale_factor)


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_scale",
                        action="store",
                        type=int,
                        default=50,
                        help="indicate user scale")

    parser.add_argument("--product_scale",
                        action="store",
                        type=int,
                        default=100,
                        help="indicate product scale")

    parser.add_argument("--purchase_scale",
                        action="store",
                        type=int,
                        default=1000000,
                        help="indicate purchase scale")

    parser.add_argument("--observation_scale",
                        action="store",
                        type=int,
                        default=1000000,
                        help="indicate observation scale")

    args = parser.parse_args()
    user_scale_factor = args.user_scale
    purchase_scale_factor = args.purchase_scale
    observation_scale_factor = args.observation_scale
    product_scale = args.product_scale

    return user_scale_factor, purchase_scale_factor, observation_scale_factor, product_scale


def write_csv(outout_path, header, data):
    with open(outout_path, 'w+', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        for item in data:
            # write the data
            writer.writerow(item)


def generate_user_profile(user_scale):
    us_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

    user_ids = np.arange(user_scale)
    user_genders = np.random.choice([0, 1, 2], size=user_scale)
    user_ages = np.random.choice(np.arange(15, 100), size=user_scale)
    gift_card_balance = np.random.choice([0, 100, 200, 300, 400, 500], size=user_scale)
    number_credit_card = np.random.choice(np.arange(0, 10), size=user_scale)
    user_states = np.random.choice(us_states, size=user_scale)
    tax_rate = np.random.choice(np.arange(0, 1.3, 0.05), size=user_scale)

    user_profile_data = list()

    for i in user_ids:
        data_item = list()
        data_item.append(i)
        data_item.append(user_genders[i])
        data_item.append(user_ages[i])
        data_item.append(gift_card_balance[i])
        data_item.append(number_credit_card[i])
        data_item.append(user_states[i])
        data_item.append(round(tax_rate[i], 1))
        user_profile_data.append(data_item)

    user_profile_header = ['user_id', 'gender', 'age', 'gift_card_balance', 'number_of_credit_cards', 'state', 'tax_rate']

    return user_profile_header, user_profile_data


def generate_purchase_history(user_scale, purchase_scale):
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2021, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    user_ids_purchase = np.random.choice(np.arange(user_scale), size=purchase_scale)
    purchase_amounts = np.random.choice(np.arange(0, 1000, 0.01), size=purchase_scale)

    purchase_history_data = list()

    for i in np.arange(purchase_scale):
        data_item = list()
        data_item.append(user_ids_purchase[i])
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        data_item.append(random_date)
        data_item.append(round(purchase_amounts[i], 2))
        purchase_history_data.append(data_item)

    user_purchase_history_header = ['user_id', 'purchase_date', 'purchase_amount']

    return user_purchase_history_header, purchase_history_data


def generate_observation(user_scale, product_scale, observation_scale):
    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2022, 6, 1)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    user_ids_observation = np.random.choice(np.arange(user_scale), size=observation_scale)
    prod_ids_observation = np.random.choice(np.arange(product_scale), size=observation_scale)
    rating_observation = np.random.choice(np.arange(1, 6), size=observation_scale)

    observation_data = list()

    for i in np.arange(observation_scale):
        data_item = list()
        data_item.append(user_ids_observation[i])
        data_item.append(prod_ids_observation[i])
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        data_item.append(random_date)
        data_item.append(rating_observation[i])
        observation_data.append(data_item)

    observation_header = ['user_id', 'product_id', 'event_timestamp', 'product_rating']

    return observation_header, observation_data


def main():
    user_scale, purchase_scale, observation_scale, product_scale = get_argument()
    user_profile_header, user_profile = generate_user_profile(user_scale)
    write_csv("user_profile_{}.csv".format(convert_km(user_scale)), user_profile_header, user_profile)

    purchase_history_header, purchase_history = generate_purchase_history(user_scale, purchase_scale)
    write_csv("purchase_history_{}.csv".format(convert_km(purchase_scale)), purchase_history_header, purchase_history)

    observation_header, observation_history = generate_observation(user_scale, product_scale, observation_scale)
    write_csv("observation_{}.csv".format(convert_km(observation_scale)), observation_header, observation_history)


if __name__ == "__main__":
    main()
