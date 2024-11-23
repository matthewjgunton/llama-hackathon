## goal:
- help people better connect with people in their immediate vicinity
- hard of hearing: I have trouble hearing people
- different language: I have trouble understanding people & having them understand me

## High-Level Idea
- Translated in near-real time
- save the discussion
- (speaking) T5-TTS voice cloning

## game plan
- translate what the other user is saying
- text is translated 
- fact-checked by ollama
    - key terms / summarizations are displayed for user

## test:
- spanish conversation
All English text:
```
**Customer:** Hi, I'm calling about my recent statement. I noticed that I was charged a fee for an overdraft.

**Bank Representative:** Yes, let me check on that for you. Can you confirm your account number?

**Customer:** It's 1234567890.

**Bank Representative:** Okay, I've located the transaction. The overdraft occurred on [date] when the balance in your
account dropped below $50. We charged a fee of $30 due to insufficient funds.

**Customer:** But I'm certain I paid the bill within the 24-hour grace period. I usually pay my bills on time and never had
an issue like this before.

**Bank Representative:** (pause) Let me see if there's any note about this... Ah, yes. It looks like our system processed
the overdraft fee even though you did make a deposit shortly after the withdrawal. The deposit cleared within the 24-hour
window.

**Customer:** Exactly! That must be why I didn't notice it right away.

**Bank Representative:** (apologetically) Yes, and that's not how we should have handled this situation. Let me go ahead and
refund the overdraft fee to your account.

**Customer:** Thank you so much for looking into this. I really appreciate it.

**Bank Representative:** You're welcome! I'm sorry again for the mistake. If you experience any other issues or have
questions, please don't hesitate to contact us.
```

All Spanish text: 
```
**Customer:** Hola, estoy llamando sobre mi reciente estado de cuenta. Me di cuenta de que me cobraron una tarifa por un
descubierto.

**Bank Representative:** Sí, veámoslo. ¿Puede confirmar su número de cuenta?

**Customer:** Es 1234567890.

**Bank Representative:** Bueno, he ubicado la transacción. El descubierto ocurrió el [fecha] cuando el saldo en su cuenta
cayó por debajo de $50. Le cobramos una tarifa de $30 debido a fondos insuficientes.

**Customer:** Pero estoy seguro de que pagué la factura dentro del plazo de gracia de 24 horas. Pago mis facturas siempre a
tiempo y nunca tuve un problema como este antes.

**Bank Representative:** (pausa) Vea si hay algún comentario sobre esto... Ah, sí. Parece ser que nuestro sistema procesó la
tarifa por descubierto aunque depositó una cantidad poco después del retiro. La depósito se cobró dentro de los 24 horas.

**Customer:** Exactamente! Eso debe ser por qué no me lo di cuenta al principio.

**Bank Representative:** (con pesar) Sí, y eso no es cómo debimos haber manejado esta situación. Vayamos a reflejar la
tarifa de descubierto en su cuenta.

**Customer:** Muchas gracias por investigarlo. Me aprecio mucho.

**Bank Representative:** De nada! Lo siento mucho por el error. Si experimenta algún otro problema o tiene alguna pregunta,
no dude en hacérmelo saber.
```


## scams people run into
- **phrasing here is key**
- 




I talk >>> sent over internet >>> model processes >>> sends back