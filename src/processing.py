import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from phik import phik_matrix


class DataProcessor:
    # Класс для загрузки, предобработки и анализа данных Яндекс Афиши.

    def __init__(self):
        load_dotenv()
        
        self.db_config = {
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PWD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'dbname': os.getenv('DB_NAME'),
        }
        
        self.raw_data = None
        self.tenge_rates = None
        self.processed_data = None
        self.user_profiles = None
        
        # Переменные для хранения метрик отчёта
        self.mean_return_rate = 0.0
        self.top_segments = []

    def load_data_from_db(self):
        # Загрузка сырых данных из PostgreSQL.# 
        print("Загружаем данные из базы...")
        connection_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        engine = create_engine(connection_string, connect_args={"sslmode": "require"})
        
        query = '''
        SELECT 
            p.user_id, p.device_type_canonical, p.order_id, p.created_dt_msk AS order_dt, p.created_ts_msk AS order_ts,
            p.currency_code, p.revenue, p.tickets_count,
            (p.created_dt_msk::date - LAG(p.created_dt_msk::date) OVER (PARTITION BY p.user_id ORDER BY p.created_dt_msk)) AS days_since_prev,
            e.event_id, e.event_name_code AS event_name, e.event_type_main, p.service_name, r.region_name, c.city_name
        FROM afisha.purchases p
        JOIN afisha.events e ON p.event_id = e.event_id
        JOIN afisha.city c ON e.city_id = c.city_id
        JOIN afisha.regions r ON c.region_id = r.region_id
        WHERE p.device_type_canonical IN ('mobile', 'desktop') 
          AND e.event_type_main != 'фильм'
        ORDER BY p.user_id ASC, p.created_dt_msk ASC
        '''
        self.raw_data = pd.read_sql_query(query, con=engine)
        print(f"Данные успешно загружены! Строк: {self.raw_data.shape[0]}")
        return self.raw_data

    def load_tenge_rates(self, filepath='data/final_tickets_tenge_df.csv'):
        # Загрузка курсов тенге из CSV.# 
        print("Загружаем курсы тенге.")
        self.tenge_rates = pd.read_csv(filepath, sep=',')
        self.tenge_rates['data'] = pd.to_datetime(self.tenge_rates['data']).dt.date
        print(f"Курсы загружены. Строк: {self.tenge_rates.shape[0]}")
        print()
        return self.tenge_rates

    def preprocess(self):
        # Приведение типов, конвертация валют, нормализация, фильтрация выбросов и дублей.# 
        print("Начинаем предобработку данных.")
        df = self.raw_data.copy()
        
        # 1. Конвертация валют
        df['order_dt'] = pd.to_datetime(df['order_dt']).dt.date
        df = df.merge(self.tenge_rates[['data', 'curs']], left_on='order_dt', right_on='data', how='left')
        df['revenue_rub'] = df['revenue']
        kzt_mask = df['currency_code'] == 'kzt'
        df.loc[kzt_mask, 'revenue_rub'] = df.loc[kzt_mask, 'revenue'] * (df.loc[kzt_mask, 'curs'] / 100)
        df = df.drop(columns=['data'])
        
        # 2. Нормализация текста
        df['device_type_canonical'] = df['device_type_canonical'].astype(str).str.strip().str.lower()
        df['currency_code'] = df['currency_code'].astype(str).str.strip().str.lower()
        for col in ['event_type_main', 'service_name', 'region_name', 'city_name', 'event_name']:
            df[col] = df[col].astype(str).str.strip().str.title()
            
        df.replace({'nan': np.nan, 'н/д': np.nan, 'неизвестно': np.nan, 'unknown': np.nan, 'None': np.nan}, inplace=True)
        
        # 3. Оптимизация типов данных (downcast)
        df['tickets_count'] = pd.to_numeric(df['tickets_count'], downcast='integer')
        df['revenue'] = pd.to_numeric(df['revenue'], downcast='float')
        df['revenue_rub'] = pd.to_numeric(df['revenue_rub'], downcast='float')
        df['days_since_prev'] = pd.to_numeric(df['days_since_prev'], downcast='float')
        
        # 4. Фильтрация выбросов (99-й перцентиль и >= 0)
        q99_rev = df['revenue_rub'].quantile(0.99)
        q99_tix = df['tickets_count'].quantile(0.99)
        df = df[(df['revenue_rub'] >= 0) & (df['revenue_rub'] <= q99_rev) & (df['tickets_count'] <= q99_tix)].copy()
        
        # 5. Удаление дубликатов
        content_cols = ['user_id', 'order_ts', 'event_id', 'revenue_rub', 'tickets_count', 'device_type_canonical', 'service_name']
        df = df.drop_duplicates(subset=['order_id'], keep='first')
        df = df.drop_duplicates(subset=content_cols, keep='first')
        df = df.sort_values(by=['user_id', 'order_ts']).reset_index(drop=True)
        
        # 6. Перевод в category для экономии памяти
        cat_cols = ['device_type_canonical', 'currency_code', 'event_type_main', 'service_name', 'region_name', 'city_name']
        for col in cat_cols:
            df[col] = df[col].astype('category')
            
        self.processed_data = df
        print(f"Предобработка завершена! Итоговый размер: {self.processed_data.shape[0]} строк.")
        print()
        return self.processed_data

    def build_user_profiles(self):
        # Создание и очистка профилей пользователей.# 
        print("Строим профили пользователей.")
        df = self.processed_data.copy()
        
        # Данные первого заказа
        first_order = df.sort_values('order_ts').groupby('user_id').first().reset_index()[
            ['user_id', 'order_dt', 'device_type_canonical', 'region_name', 'event_type_main']
        ]
        first_order.rename(columns={
            'order_dt': 'first_order_dt', 'device_type_canonical': 'first_device',
            'region_name': 'first_region', 'event_type_main': 'first_event_type'
        }, inplace=True)
        
        # Агрегация по пользователю
        user_profile = df.groupby('user_id').agg(
            last_order_dt=('order_dt', 'max'),
            total_orders=('order_id', 'nunique'),
            avg_revenue_rub=('revenue_rub', 'mean'),
            avg_days_between=('days_since_prev', 'mean')
        ).reset_index()
        
        user_profile = user_profile.merge(first_order, on='user_id', how='left')
        user_profile['is_two'] = (user_profile['total_orders'] >= 2).astype(int)
        user_profile['first_day_of_week'] = pd.to_datetime(user_profile['first_order_dt']).dt.dayofweek
        day_names = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}
        user_profile['first_day_name'] = user_profile['first_day_of_week'].map(day_names)
        
        # Фильтрация аномальных профилей (>99 перцентиля заказов или <= 0 выручка)
        q99_orders = user_profile['total_orders'].quantile(0.99)
        self.user_profiles = user_profile[
            (user_profile['total_orders'] <= q99_orders) & (user_profile['avg_revenue_rub'] > 0)
        ].copy()
        
        print(f"Построено {len(self.user_profiles)} чистых профилей пользователей.")
        print()
        return self.user_profiles

    def calculate_retention(self):
        # Расчёт Retention по сегментам и проверка гипотез.
        print("Рассчитываем Retention по сегментам.")
        self.mean_return_rate = self.user_profiles['is_two'].mean()
        
        # Статистика по типу мероприятия
        self.event_stats = self.user_profiles.groupby('first_event_type').agg(
            total_users=('user_id', 'count'), returned_users=('is_two', 'sum')
        ).reset_index()
        self.event_stats['return_rate'] = self.event_stats['returned_users'] / self.event_stats['total_users']
        
        # Статистика по устройству
        self.device_stats = self.user_profiles.groupby('first_device').agg(
            total_users=('user_id', 'count'), returned_users=('is_two', 'sum')
        ).reset_index()
        self.device_stats['return_rate'] = self.device_stats['returned_users'] / self.device_stats['total_users']
        # Статистика по регионам
        self.region_stats = self.user_profiles.groupby('first_region').agg(
            total_users=('user_id', 'count'), returned_users=('is_two', 'sum')
        ).reset_index()
        self.region_stats = self.region_stats.sort_values('total_users', ascending=False).head(10)
        self.region_stats['return_rate'] = self.region_stats['returned_users'] / self.region_stats['total_users']
        
        # Статистика по дням недели
        self.dow_stats = self.user_profiles.groupby('first_day_name').agg(
            total_users=('user_id', 'count'), returned_users=('is_two', 'sum')
        ).reset_index()
        day_order = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        self.dow_stats['first_day_name'] = pd.Categorical(self.dow_stats['first_day_name'], categories=day_order, ordered=True)
        self.dow_stats = self.dow_stats.sort_values('first_day_name')
        self.dow_stats['return_rate'] = self.dow_stats['returned_users'] / self.dow_stats['total_users']
        print("Расчёты удержания завершены.")

    def correlation_analysis(self):
        # Корреляционный анализ phi_k по сегментам.
        print("Проводим корреляционный анализ.")
        
        df_corr = self.user_profiles.copy()
        
        def create_order_segment(n_orders):
            if n_orders == 1: return '1 заказ'
            elif n_orders == 2: return '2 заказа'
            elif 3 <= n_orders <= 4: return '3-4 заказа'
            else: return '5+ заказов'
            
        df_corr['order_segment'] = df_corr['total_orders'].apply(create_order_segment)
        
        features = ['avg_revenue_rub', 'avg_days_between', 'first_device', 'first_region', 'first_event_type']
        
        # Вывод по сегменту 5+ заказов, с самой сильной связью
        segment_data = df_corr[df_corr['order_segment'] == '5+ заказов']
        corr_matrix = phik_matrix(
            segment_data[features + ['total_orders']], 
            interval_cols=['avg_revenue_rub', 'avg_days_between', 'total_orders']
        )
        self.corr_5plus = corr_matrix['total_orders'].drop('total_orders').sort_values(ascending=False)
        print("Корреляционный анализ завершён.")

    def check_hypotheses(self):
        # Проверка двух продуктовых гипотез.
        print("Проверка продуктовых гипотез.")
        
        # Гипотеза 1: Спорт vs Концерты
        sport_return = self.user_profiles[self.user_profiles['first_event_type'] == 'Спорт']['is_two'].mean()
        concert_return = self.user_profiles[self.user_profiles['first_event_type'] == 'Концерты']['is_two'].mean()
        sport_count = self.user_profiles[self.user_profiles['first_event_type'] == 'Спорт'].shape[0]
        concert_count = self.user_profiles[self.user_profiles['first_event_type'] == 'Концерты'].shape[0]
        
        self.hypothesis_1 = {
            'sport_return': sport_return,
            'concert_return': concert_return,
            'sport_count': sport_count,
            'concert_count': concert_count
        }
        print(f"   Гипотеза 1 (На Спорте возвратов больше, чем на Концерты): Спорт — {sport_return:.1%} ({sport_count} чел.), Концерты — {concert_return:.1%} ({concert_count} чел.)")
        
        # Гипотеза 2: ТОП-3 региона vs остальные
        top3_regions = self.user_profiles['first_region'].value_counts().head(3).index.tolist()
        self.user_profiles['is_top3_region'] = self.user_profiles['first_region'].isin(top3_regions)
        
        top3_return = self.user_profiles[self.user_profiles['is_top3_region']]['is_two'].mean()
        other_return = self.user_profiles[~self.user_profiles['is_top3_region']]['is_two'].mean()
        top3_count = self.user_profiles[self.user_profiles['is_top3_region']].shape[0]
        other_count = self.user_profiles[~self.user_profiles['is_top3_region']].shape[0]
        other_n_regions = self.user_profiles[~self.user_profiles['is_top3_region']]['first_region'].nunique()
        
        self.hypothesis_2 = {
            'top3_return': top3_return,
            'other_return': other_return,
            'top3_count': top3_count,
            'other_count': other_count,
            'other_n_regions': other_n_regions,
            'top3_regions': top3_regions
        }
        print(f"   Гипотеза 2 (В ТОП-3 регионах по количеству пользователей доля возвратов выше): ТОП-3 ({', '.join(top3_regions)}) — {top3_return:.1%} ({top3_count} чел.), Остальные ({other_n_regions} регионов) — {other_return:.1%} ({other_count} чел.)")

    def generate_report(self):
        # Вывод краткого итогового отчёта в терминал.
        print()
        print("ОБЩИЙ ВЫВОД")
        print()
        
        print(f"ДАННЫЕ: Итоговая чистая выборка: {len(self.user_profiles)} пользователей и {len(self.processed_data)} заказов.")
        print(f"   Доля вернувшихся пользователей (2+ заказа): {self.mean_return_rate:.1%}")
        
        print("ТОЧКИ ВХОДА: Лучшие показатели удержания у пользователей, чей первый заказ был:")
        top_event = self.event_stats.loc[self.event_stats['return_rate'].idxmax()]
        top_device = self.device_stats.loc[self.device_stats['return_rate'].idxmax()]
        print(f" - Мероприятие: {top_event['first_event_type']} ({top_event['return_rate']:.1%})")
        print(f" - Устройство: {top_device['first_device']} ({top_device['return_rate']:.1%})")
        
        print("КОРРЕЛЯЦИИ: В сегменте активных пользователей (5+ заказов) выявлена сильная связь")
        if self.corr_5plus is not None:
            top_corr = self.corr_5plus.idxmax()
            print(f"   между регулярностью покупок и общим числом заказов (признак '{top_corr}', phi_k ≈ {self.corr_5plus.max():.3f}).")
        else:
            print("   между регулярностью покупок и общим числом заказов.")
        print()
        print("АНАЛИЗ ПО РЕГИОНАМ И ДНЯМ НЕДЕЛИ:")
        top_region = self.region_stats.loc[self.region_stats['return_rate'].idxmax()]
        print(f"- Лучший регион: {top_region['first_region']} ({top_region['return_rate']:.1%}, {int(top_region['total_users'])} чел.)")
        
        top_day = self.dow_stats.loc[self.dow_stats['return_rate'].idxmax()]
        print(f"- Лучший день недели: {top_day['first_day_name']} ({top_day['return_rate']:.1%}, {int(top_day['total_users'])} чел.)")
        print()
        print("ПРОВЕРКА ГИПОТЕЗ о доле возвратов:")
        print(f"- Гипотеза 1 (Спорт vs Концерты): {'Подтвердилась' if self.hypothesis_1['sport_return'] > self.hypothesis_1['concert_return'] else 'Не подтвердилась'}")
        print(f"- Гипотеза 2 (ТОП-3 региона): {'Подтвердилась' if self.hypothesis_2['top3_return'] > self.hypothesis_2['other_return'] else 'Не подтвердилась'}")    
        print()
        print("РЕКОМЕНДАЦИИ:")
        print("   - Внедрить персональные скидки на второй заказ сразу после первой покупки.")
        print("   - Увеличить продвижение категорий 'Выставки' и 'Театр' как каналов привлечения.")
        print("   - Проанализировать и улучшить пользовательский путь в мобильном приложении.")
        print("   - Для сегмента 5+ заказов внедрить накопительную систему бонусов.")
        print()

    def run_pipeline(self):
        self.load_data_from_db()
        self.load_tenge_rates()
        self.preprocess()
        self.build_user_profiles()
        self.calculate_retention()
        self.correlation_analysis()
        self.check_hypotheses()
        self.generate_report()