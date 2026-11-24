import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api",
  timeout: 180000,
});

export default client;

// Keep existing pages compatible with their parsed-data, fetch-style helper.
export async function api(path, { body, ...options } = {}) {
  try {
    const response = await client.request({
      ...options,
      url: path,
      data: body,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
    return response.data;
  } catch (error) {
    const detail = error.response?.data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => `${item.loc.slice(1).join(".")}: ${item.msg}`).join("; ")
          : error.code === "ECONNABORTED"
            ? "The request timed out after 3 minutes. Please try again."
            : error.message || "Request failed";
    throw new Error(message, { cause: error });
  }
}
