# Database Structure

This document provides an overview of the database structure, including the tables and their purposes.

## Tables

### Spatial Tables
- **geography_columns**: Stores information about geography columns.
- **geometry_columns**: Stores information about geometry columns.
- **spatial_ref_sys**: Stores information about spatial reference systems.

### Deals and Promotions
- **banners**: Stores information about promotional banners.
- **bulk_deals**: Stores information about bulk deals.
- **group_deals**: Stores information about group deals.
- **single_deals**: Stores information about single deals.

### Carts and Orders
- **carts**: Stores information about shopping carts.
- **groups_carts**: Stores information about group carts.
- **group_cart_variations**: Stores variations of group carts.
- **orders**: Stores information about orders.
- **personal_cart_items**: Stores items in personal carts.

### Products and Categories
- **categories**: Stores information about product categories.
- **category_localizations**: Stores localized information for categories.
- **attributes**: Stores product attributes.
- **attribute_values**: Stores values for product attributes.
- **product_names**: Stores product names.
- **product_name_localizations**: Stores localized product names.
- **product_variations**: Stores variations of products.
- **product_variation_prices**: Stores prices for product variations.
- **product_variation_attributes**: Stores attributes for product variations.
- **product_variation_stocks**: Stores stock information for product variations.
- **product_stocks**: Stores stock information for products.
- **product_ratings**: Stores ratings for products.

### Users and Drivers
- **users**: Stores information about users.
- **drivers**: Stores information about drivers.
- **driver_rating**: Stores ratings for drivers.

### Payments and Rewards
- **payment_methods**: Stores information about payment methods.
- **rewards**: Stores information about rewards.

### Notifications and Configurations
- **notification_template**: Stores templates for notifications.
- **configs**: Stores configuration settings.

### Miscellaneous
- **casbin_rule**: Stores access control rules.
- **countries**: Stores information about countries.
- **devices**: Stores information about devices.
- **delivery_tracks**: Stores information about delivery tracks.
- **routes**: Stores information about routes.