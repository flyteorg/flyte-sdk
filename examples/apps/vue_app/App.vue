<script setup>
import { ref, computed } from 'vue'

const input = ref('')
const items = ref([])

const remaining = computed(() => items.value.filter(i => !i.done).length)

function add() {
  if (!input.value.trim()) return
  items.value.push({ id: Date.now(), text: input.value.trim(), done: false })
  input.value = ''
}
</script>

<template>
  <div class="app">
    <header>
      <h1>Todos</h1>
      <span v-if="items.length" class="badge">{{ remaining }} left</span>
    </header>

    <div class="input-row">
      <input
        v-model="input"
        @keyup.enter="add"
        placeholder="What needs to be done?"
        autofocus
      />
      <button @click="add">Add</button>
    </div>

    <ul v-if="items.length">
      <li v-for="item in items" :key="item.id" :class="{ done: item.done }">
        <input type="checkbox" v-model="item.done" />
        <span>{{ item.text }}</span>
      </li>
    </ul>

    <p v-else class="empty">No tasks yet — add one above.</p>
  </div>
</template>

<style scoped>
.app {
  max-width: 480px;
  margin: 5rem auto;
  padding: 2rem;
  font-family: system-ui, sans-serif;
  border-radius: 12px;
  box-shadow: 0 4px 32px rgba(0, 0, 0, 0.08);
}

header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

h1 {
  margin: 0;
  font-size: 1.75rem;
}

.badge {
  background: #6366f1;
  color: white;
  font-size: 0.75rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}

.input-row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

input:not([type='checkbox']) {
  flex: 1;
  padding: 0.6rem 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  outline: none;
}

input:not([type='checkbox']):focus {
  border-color: #6366f1;
}

button {
  padding: 0.6rem 1.2rem;
  background: #6366f1;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
}

button:hover {
  background: #4f46e5;
}

ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

li {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid #f1f5f9;
}

li.done span {
  text-decoration: line-through;
  opacity: 0.45;
}

.empty {
  color: #94a3b8;
  text-align: center;
  padding: 2rem 0;
}
</style>
