/**
 * Order -> feature vector, encoded exactly the way training encoded it.
 *
 * The category maps are exported from model/train.py rather than rebuilt here,
 * so an unknown category is a hard error instead of a silent zero. A silent
 * zero would mean "Accessories" and would produce a confident, wrong number.
 */

export class ValidationError extends Error {
  constructor(field, message, allowed) {
    super(message);
    this.name = "ValidationError";
    this.field = field;
    this.allowed = allowed;
  }
}

/** Shipping mode fixes the scheduled window in this dataset. */
export const CANONICAL_DAYS = {
  "Same Day": 0,
  "First Class": 1,
  "Second Class": 2,
  "Standard Class": 4,
};

const NUMERIC_BOUNDS = {
  "Days for shipment (scheduled)": [0, 6],
  "Order Item Quantity": [1, 5],
  "Order Item Product Price": [0, 5000],
  "Order Item Discount Rate": [0, 0.25],
};

/**
 * @returns {{x: number[], resolved: object, warnings: string[]}}
 */
export function buildVector(encoders, order) {
  if (!order || typeof order !== "object") {
    throw new ValidationError("order", "order must be an object");
  }
  const warnings = [];

  const shippingMode = requireCategory(encoders, "Shipping Mode", order.shipping_mode);
  const market = requireCategory(encoders, "Market", order.market);
  const region = requireCategory(encoders, "Order Region", order.region);
  const category = requireCategory(encoders, "Category Name", order.category);
  const paymentType = requireCategory(encoders, "Type", order.payment_type);

  // The planner normally just picks a shipping mode; the window follows.
  let scheduledDays = order.scheduled_days;
  if (scheduledDays === undefined || scheduledDays === null) {
    scheduledDays = CANONICAL_DAYS[shippingMode];
  } else {
    scheduledDays = requireNumber("scheduled_days", scheduledDays,
      NUMERIC_BOUNDS["Days for shipment (scheduled)"]);
    if (scheduledDays !== CANONICAL_DAYS[shippingMode]) {
      warnings.push(
        `Bu veri setinde "${shippingMode}" her zaman ` +
        `${CANONICAL_DAYS[shippingMode]} günlük pencereyle geliyor. ` +
        `${scheduledDays} gün eğitim dağılımının dışında — model burada ` +
        `tahmin ediyor, ölçmüyor.`
      );
    }
  }

  const quantity = requireNumber("quantity", fallback(order.quantity, 1),
    NUMERIC_BOUNDS["Order Item Quantity"]);
  const unitPrice = requireNumber("unit_price", fallback(order.unit_price, 100),
    NUMERIC_BOUNDS["Order Item Product Price"]);
  const discountRate = requireNumber("discount_rate", fallback(order.discount_rate, 0),
    NUMERIC_BOUNDS["Order Item Discount Rate"]);

  const { month, dayOfWeek } = parseOrderDate(order.order_date);

  const named = {
    "Days for shipment (scheduled)": scheduledDays,
    "Order Item Quantity": quantity,
    "Order Item Product Price": unitPrice,
    "Order Item Discount Rate": discountRate,
    order_month: month,
    order_day_of_week: dayOfWeek,
    "Shipping Mode": encoders.categorical["Shipping Mode"][shippingMode],
    Market: encoders.categorical["Market"][market],
    "Order Region": encoders.categorical["Order Region"][region],
    "Category Name": encoders.categorical["Category Name"][category],
    Type: encoders.categorical["Type"][paymentType],
  };

  const x = encoders.order.map((name) => {
    if (!(name in named)) throw new ValidationError(name, `no value built for ${name}`);
    return named[name];
  });

  return {
    x,
    resolved: {
      shipping_mode: shippingMode,
      scheduled_days: scheduledDays,
      market,
      region,
      category,
      payment_type: paymentType,
      quantity,
      unit_price: unitPrice,
      discount_rate: discountRate,
      order_month: month,
      order_day_of_week: dayOfWeek,
    },
    warnings,
  };
}

/** Regions belong to markets; offering the wrong pair is a common mistake. */
export function regionsByMarket(encoders, marketsToRegions) {
  return marketsToRegions ?? null;
}

export function allowedValues(encoders) {
  const out = {};
  for (const [field, map] of Object.entries(encoders.categorical)) {
    out[field] = Object.keys(map);
  }
  return out;
}

function requireCategory(encoders, field, value) {
  const map = encoders.categorical[field];
  if (value === undefined || value === null || value === "") {
    throw new ValidationError(field, `${field} zorunlu`, Object.keys(map));
  }
  const str = String(value);
  if (str in map) return str;

  // Trailing spaces survive in this dataset ("South of  USA ") — match on a
  // normalised key before giving up, so a sane caller is not punished for it.
  const norm = (s) => s.replace(/\s+/g, " ").trim().toLowerCase();
  const hit = Object.keys(map).find((k) => norm(k) === norm(str));
  if (hit) return hit;

  throw new ValidationError(field, `"${str}" ${field} için geçerli değil`, Object.keys(map));
}

function requireNumber(field, value, [lo, hi]) {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) throw new ValidationError(field, `${field} sayı olmalı`);
  if (n < lo || n > hi) {
    throw new ValidationError(field, `${field} ${lo} ile ${hi} arasında olmalı (gelen: ${n})`);
  }
  return n;
}

function fallback(value, dflt) {
  return value === undefined || value === null || value === "" ? dflt : value;
}

function parseOrderDate(value) {
  if (!value) {
    // No date given: use a neutral mid-week, mid-year point rather than "now",
    // so the same request always returns the same number.
    return { month: 6, dayOfWeek: 2 };
  }
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    throw new ValidationError("order_date", `order_date okunamadı: "${value}"`);
  }
  // Training used pandas: Monday = 0 ... Sunday = 6.
  return { month: d.getUTCMonth() + 1, dayOfWeek: (d.getUTCDay() + 6) % 7 };
}
